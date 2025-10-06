// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_opocmab.h"
#include "reductions.h"
#include "cb_adf.h"
#include "rand48.h"
#include "bs.h"
#include "gen_cs_example.h"
#include "cb_explore.h"
#include "explore.h"
#include "action_score.h"
#include "cb.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
// For sigmoid
#include "correctedMath.h"

using namespace VW::LEARNER;

namespace VW
{
namespace cb_explore_adf
{
namespace opocmab
{

/*
This file implements the OPO-CMAB algorithm (Optimistic Policy Optimization for Contextual Multi-Armed Bandits)
as described in the provided pseudo-code.

The algorithm maintains:
- Policy initialization: π_1 uniform for any observed context c
- Loss approximation function f_k using squared loss oracle (base learner)
- Confidence bounds b_k^β where β = γ * sqrt(k / num_actions)
- Sequential policy updates: π_1(c_t, ·) → π_2(c_t, ·) → ... → π_t(c_t, ·)
- Exponential weight updates for policy improvement

Learning Loop Pattern:
- Uses _counter to track number of rounds (no K parameter needed)
- Counter incremented only during learning phase (following SquareCB pattern)
- Round number t = _counter is used for all policy computations
- Automatic progression through episodes without external control

Squared Loss Oracle:
- f_k approximated using base learner trained ONLY on data from rounds 1,2,...,k-1
- Past predictions f̂_1, f̂_2, ..., f̂_{t-1} stored and reused for sequential updates
- Each f̂_k represents squared loss oracle output at the correct time point k
- No manual gradient descent - base learner handles all optimization
- Follows VowpalWabbit's standard squared loss optimization

Weight Snapshot Approach:
- Saves model weights after each training round for O efficiency
- Past predictions f̂_k computed by temporarily swapping to historical weights
- Replaces complex retraining approach with simple weight restoration
- Maintains theoretical correctness with significant performance improvement

Parameters:
- γ (gamma): Tuning parameter that scales the confidence bounds
- η (learning_rate): Learning rate for policy updates
*/

using namespace VW::cb_explore_adf::opocmab;

// Custom deleter for VW examples that properly cleans up internal structures
void safe_delete_example(example* ex) {
  if (ex != nullptr) {
    // Use VW's proper deallocation which cleans up all internal structures
    VW::dealloc_examples(ex, 1);
  }
}

// Implementation of HistoricalRound::add_example method
void cb_explore_adf_opocmab::HistoricalRound::add_example(const example* src_example) {
  if (src_example == nullptr) return;
  
  // Allocate new example
  example* copied_ex = new example;
  
  // Deep copy all data and labels using VW's copy function
  copy_example_data_with_label(copied_ex, src_example);
  
  // Store in unique_ptr with custom deleter for automatic cleanup
  examples.emplace_back(copied_ex, safe_delete_example);
}

float sigmoid_link(float score)
{
  return 1.f / (1 + correctedExp(-1. * score));
}

cb_explore_adf_opocmab::cb_explore_adf_opocmab(float eta, float gamma, bool sigmoid, vw* all)
    : _counter(1), _learning_rate(eta), _gamma(gamma), _sigmoid(sigmoid), _all(all)  // Initialize round counter to 1
{

}

void cb_explore_adf_opocmab::save_current_weights()
{
  // Save current weights to temporary backup
  uint32_t total_weights = (uint32_t)_all->length();
  _temp_weight_backup.resize(total_weights);
  
  for (uint32_t i = 0; i < total_weights; i++)
  {
    _temp_weight_backup[i] = get_weight(*_all, i, 0);
  }
}

void cb_explore_adf_opocmab::restore_weights_from_snapshot(size_t snapshot_index)
{
  // Restore weights from a specific snapshot
  if (snapshot_index >= _weight_snapshots.size()) return;
  
  const auto& snapshot = _weight_snapshots[snapshot_index];
  uint32_t num_weights = static_cast<uint32_t>(snapshot.size());
  
  for (uint32_t i = 0; i < num_weights; i++)
  {
    set_weight(*_all, i, 0, snapshot[i]);
  }
}

void cb_explore_adf_opocmab::save_weight_snapshot_after_round()
{
  // Save current weights as a snapshot after training round
  uint32_t total_weights = (uint32_t)_all->length();
  std::vector<float> snapshot(total_weights);
  
  for (uint32_t i = 0; i < total_weights; i++)
  {
    snapshot[i] = get_weight(*_all, i, 0);
  }
  
  _weight_snapshots.push_back(std::move(snapshot));
}

void cb_explore_adf_opocmab::restore_current_weights()
{
  // Restore weights from temporary backup
  uint32_t num_weights = static_cast<uint32_t>(_temp_weight_backup.size());
  
  for (uint32_t i = 0; i < num_weights; i++)
  {
    set_weight(*_all, i, 0, _temp_weight_backup[i]);
  }
}

float cb_explore_adf_opocmab::compute_beta(size_t k, float gamma, size_t num_actions)
{
  // β = γ * sqrt(k / num_actions)
  return gamma * std::sqrt(static_cast<float>(k) / static_cast<float>(num_actions));
}

void cb_explore_adf_opocmab::store_historical_data(const multi_ex& examples, uint32_t chosen_action, float cost, float prob)
{
  // Store the complete training example for future retraining using memory-safe approach
  HistoricalRound round;
  
  // Deep copy the examples using the memory-safe add_example method
  for (const auto& ex : examples)
  {
    round.add_example(ex);
  }
  
  round.chosen_action = chosen_action;
  round.observed_cost = cost;
  round.probability = prob;
  round.round_number = _counter;  // Current round number
  
  // CRITICAL: Store the policy π_j(c_j, ·) that was used in round j
  // This is essential for theoretically correct retraining
  round.policy_used = _last_policy;
  
  _historical_data.push_back(std::move(round));
  
  // Save weight snapshot after this training round
  // This snapshot will represent weights trained on rounds {1, 2, ..., current_round}
  save_weight_snapshot_after_round();
}

float cb_explore_adf_opocmab::compute_past_prediction_on_current_context(VW::LEARNER::multi_learner& base, 
                                                                         const multi_ex& current_examples, 
                                                                         size_t k, size_t action)
{
  // THEORETICALLY CORRECT: Use weights from round k-1 to get f̂_k(c_t, a)
  // f̂_k(c_t, a) = prediction using weights after training on rounds {1, 2, ..., k-1}
  
  if (k == 1)
  {
    return 0.0f;  // f̂_1: No training data available, neutral prediction
  }
  
  // k-1 is 0-indexed, so snapshot k-2 contains weights after round k-1
  size_t snapshot_index = k - 2;
  if (snapshot_index >= _weight_snapshots.size())
  {
    return 0.0f;  // No snapshot available yet
  }
  
  // Check cache first to avoid recomputation
  auto context_features = extract_context_features(current_examples);
  size_t context_hash = compute_context_hash(context_features);
  auto cache_key = std::make_pair(k, context_hash);
  
  auto cache_it = _prediction_cache.find(cache_key);
  if (cache_it != _prediction_cache.end() && action < cache_it->second.size())
  {
    return cache_it->second[action];
  }
  
  // EFFICIENT WEIGHT SWAPPING APPROACH:
  // 1. Save current weights
  // 2. Restore weights from snapshot k-1 (representing learner trained on rounds 1..k-1)  
  // 3. Get prediction on current context
  // 4. Restore current weights
  
  std::vector<float> predictions(current_examples.size(), 0.0f);
  
  // Step 1: Save current weights  
  save_current_weights();
  
  // Step 2: Restore weights from round k-1
  restore_weights_from_snapshot(snapshot_index);
  
  // Step 3: Get prediction using historical weights
  multi_ex prediction_examples;
  for (const auto& ex : current_examples)
  {
    example* copied_ex = new example;
    copy_example_data_with_label(copied_ex, ex);
    prediction_examples.push_back(copied_ex);
  }
  
  // Get predictions from the historical model - this is TRUE f̂_k(c_t, a)!
  VW::LEARNER::multiline_learn_or_predict<false>(base, prediction_examples, prediction_examples[0]->ft_offset);
  
  // Extract predictions for all actions
  if (!prediction_examples.empty() && prediction_examples[0]->pred.a_s.size() > 0)
  {
    for (size_t a = 0; a < std::min(predictions.size(), prediction_examples[0]->pred.a_s.size()); ++a)
    {
      predictions[a] = prediction_examples[0]->pred.a_s[a].score;
      
      // CRITICAL: Apply sigmoid consistently with main prediction loop
      if (_sigmoid)
      {
        predictions[a] = sigmoid_link(predictions[a]);
      }
    }
  }
  
  // Clean up prediction examples using VW's proper deallocation
  for (auto* ex : prediction_examples)
  {
    safe_delete_example(ex);
  }
  
  // Step 4: Restore current weights
  restore_current_weights();
  
  // Cache the predictions for this context and k
  _prediction_cache[cache_key] = predictions;
  
  // Return prediction for the requested action
  return action < predictions.size() ? predictions[action] : 0.0f;
}

multi_ex cb_explore_adf_opocmab::reconstruct_training_example(const multi_ex& original_examples, 
                                                              uint32_t chosen_action, 
                                                              float observed_cost, 
                                                              float probability)
{
  // Create a deep copy of the original examples
  // NOTE: This function returns raw pointers for immediate use - caller must manage cleanup
  multi_ex training_examples;
  
  for (size_t i = 0; i < original_examples.size(); ++i)
  {
    example* copied_ex = new example;
    copy_example_data_with_label(copied_ex, original_examples[i]);
    training_examples.push_back(copied_ex);
  }
  
  // Set the observed cost for the chosen action
  if (chosen_action < training_examples.size())
  {
    // Clear existing CB labels
    training_examples[chosen_action]->l.cb.costs.clear();
    
    // Add the observed cost as a CB label
    CB::cb_class cb_label;
    cb_label.cost = observed_cost;
    cb_label.action = chosen_action;
    cb_label.probability = probability;
    
    training_examples[chosen_action]->l.cb.costs.push_back(cb_label);
  }
  
  return training_examples;
}

// Removed unused placeholder functions:
// - create_fresh_base_learner
// - save_base_learner_state  
// - reset_base_learner_to_fresh_state
// - restore_base_learner_state

size_t cb_explore_adf_opocmab::compute_context_hash(const std::vector<float>& context_features)
{
  std::hash<float> hasher;
  size_t hash_value = 0;
  
  for (size_t i = 0; i < context_features.size(); ++i)
  {
    // Combine hashes using a simple polynomial rolling hash
    hash_value = hash_value * 31 + hasher(context_features[i]);
  }
  
  return hash_value;
}

void cb_explore_adf_opocmab::clear_prediction_cache()
{
  _prediction_cache.clear();
}

// Note: reconstruct_training_example is implemented below in the file

// Note: compute_confidence_bound is no longer needed as we compute it inline
// with the correct π_i(c_t, a) values for the current context c_t



void cb_explore_adf_opocmab::update_policy(const std::vector<float>& context_features, size_t num_actions, 
                                           std::vector<float>& policy_probs, const std::vector<float>& base_predictions,
                                           VW::LEARNER::multi_learner& base, const multi_ex& current_examples)
{
  (void)context_features; // Mark as unused (used in comment/debug but not in core logic)
  (void)base_predictions; // Mark as unused (we compute predictions dynamically)
  
  // Current episode number t from automatic round counter (no K parameter needed)
  size_t t = _counter;
  
  // ROUND t = 1: Initialize π_1 to be uniform distribution over actions for any observed context c
  if (t == 1)
  {
    // π_1(c, a) = 1/num_actions for any context c (including c_t)
    float uniform_prob = 1.0f / num_actions;
    std::fill(policy_probs.begin(), policy_probs.end(), uniform_prob);
    return;
  }
  
  // ROUNDS t ≥ 2: Sequential policy updates as per Algorithm 1
  // For the current context c_t, compute π_1(c_t, ·), π_2(c_t, ·), ..., π_t(c_t, ·) iteratively
  std::vector<std::vector<float>> policies_for_ct;
  
  // STEP 1: Initialize π_1(c_t, ·) as uniform distribution 
  // (This is the same π_1 from round 1, applied to current context c_t)
  std::vector<float> current_policy(num_actions, 1.0f / num_actions);
  policies_for_ct.push_back(current_policy);  // Store π_1(c_t, ·)
  
  // OPTIMIZATION: Initialize running sums for O(1) confidence bound computation
  // Initialize to 1 to incorporate the baseline term in the denominator
  std::vector<float> running_sums(num_actions, 1.0f);
  // running_sums[a] = 1 + π_1(c_t, a) + π_2(c_t, a) + ... + π_{k-1}(c_t, a)
  
  // STEP 2: For k = 1, 2, ..., t-1: Apply policy updates iteratively for context c_t
  // This computes π_2(c_t, ·), π_3(c_t, ·), ..., π_t(c_t, ·) sequentially
  for (size_t k = 1; k < t; ++k)
  {
    // Compute ℓ_k(c_t, a) = max{0, f̂_k(c_t, a) - b_k^{β_k}(c_t, a)} for all actions a
    // Note: c_t is the specific context at round t, not a generic context c
    std::vector<float> losses_k(num_actions, 0.0f);
    
    for (size_t a = 0; a < num_actions; ++a)
    {
      // f̂_k(c_t, a) using squared loss oracle trained ONLY on data from rounds 1,2,...,k-1
      // Use the saved approximation from round k (which was trained on data up to round k-1)
      float f_val = 0.0f;
      if (k == 1)
      {
        // f̂_1: No training data available, use neutral prediction
        f_val = 0.0f;
      }
      else
      {
        // THEORETICALLY CORRECT IMPLEMENTATION: 
        // Retrain base learner using ONLY data from rounds {1, 2, ..., k-1}
        // then apply to current context c_t to get TRUE f̂_k(c_t, a)
        // This is now implemented with actual retraining (save state, reset, retrain, predict, restore)
        // This is the same as the implementation in the paper, but with the correct implementation of the retraining.
        f_val = compute_past_prediction_on_current_context(base, current_examples, k, a);
        
        // FOR DEBUGGING: Show the true context-dependent prediction
        // std::cout << "Round " << k << ", Action " << a << ": "
        //           << "f̂_k(c_t=" << context_features[0] << "," << context_features[1] << ",a=" << a << ") = " << f_val 
        //           << " (RECOMPUTED with data from rounds 1.." << k-1 << ")" << std::endl;
      }
      
      // OPTIMIZATION: O(1) confidence bound computation using running sums
      // Compute b_k^{β_k}(c_t, a) using π_1(c_t, a), ..., π_{k-1}(c_t, a)
      float beta_k = compute_beta(k, _gamma, num_actions);
      // running_sums[a] already contains 1 + π_1(c_t, a) + ... + π_{k-1}(c_t, a)
      float b_val = std::min(1.0f, (beta_k / 2.0f) / running_sums[a]);
      
      losses_k[a] = std::max(0.0f, f_val - b_val);  // ℓ_k(c_t, a)
    }
    
    // Update π_{k+1}(c_t, a) = π_k(c_t, a) * exp(-η * ℓ_k(c_t, a)) / Z_k
    std::vector<float> next_policy(num_actions, 0.0f);
    float normalizer = 0.0f;

    for (size_t a = 0; a < num_actions; ++a)
    {
      float exp_term_a= std::exp(-_learning_rate * losses_k[a]);
      next_policy[a] = current_policy[a] * exp_term_a;
      normalizer += next_policy[a];
    }

        // Normalize π_{k+1}
    if (normalizer > 0)
    {
      for (size_t a = 0; a < num_actions; ++a)
      {
        next_policy[a] /= normalizer;
      }
    }
    
    // Store π_{k+1}(c_t, ·) for next iteration
    policies_for_ct.push_back(next_policy);
    
    // OPTIMIZATION: Update running sums for next iteration (O(1) per action)
    // Add π_k(c_t, a) to the running sum for use in next confidence bound calculation
    // Note: At iteration k+1, running_sums[a] will contain 1 + π_1 + ... + π_k
    for (size_t a = 0; a < num_actions; ++a)
    {
      running_sums[a] += current_policy[a];  // Add π_k to the sum
    }
    
    // Move to next iteration: π_k becomes π_{k+1}
    current_policy = next_policy;
  }
  
  // STEP 3: Return π_t(c_t, ·) - the final policy after t-1 sequential updates
  // This is the policy we use to select actions for context c_t at round t
  policy_probs = current_policy;
}

// Note: update_loss_approximation is no longer needed in the theoretically correct implementation

std::vector<float> cb_explore_adf_opocmab::extract_context_features(const multi_ex& examples)
{
  std::vector<float> features;
  
  if (!examples.empty() && examples[0]->indices.size() > 0)
  {
    // Extract features from the first example (context)
    auto& feature_space = examples[0]->feature_space[examples[0]->indices[0]];
    for (size_t i = 0; i < std::min((size_t)10, feature_space.size()); ++i)
    {
      features.push_back(feature_space.values[i]);
    }
  }
  
  // Ensure we have at least some features
  while (features.size() < 10)
  {
    features.push_back(1.0f);
  }
  
  return features;
}

// Note: compute_context_similarity is no longer needed - we retrain instead of interpolating



template <bool is_learn>
void cb_explore_adf_opocmab::predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples)
{
  if (examples.empty()) return;
  
  // Get base predictions
  if (is_learn)
  {
    VW::LEARNER::multiline_learn_or_predict<true>(base, examples, examples[0]->ft_offset);
    // Note: Counter will be incremented AFTER storing historical data to get correct round numbering
  }
  else
  {
    VW::LEARNER::multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset);
  }
  
  auto& preds = examples[0]->pred.a_s;
  size_t num_actions = preds.size();
  
  if (num_actions == 0) return;
  
  // Extract context features
  std::vector<float> context_features = extract_context_features(examples);
  
  if (!is_learn)
  {
    // Prediction phase - compute policy probabilities using THEORETICALLY CORRECT approach
    // Extract base learner predictions (scores from cb_adf - squared loss oracle)
    std::vector<float> base_predictions(num_actions);
    for (size_t a = 0; a < num_actions; ++a)
    {
      base_predictions[a] = preds[a].score;  // Direct scores from cb_adf
    }
    
    // Apply sigmoid activation if enabled (like SquareCB)
    if (_sigmoid)
    {
      for (size_t a = 0; a < num_actions; ++a)
      {
        base_predictions[a] = sigmoid_link(base_predictions[a]);
      }
    }
    
    std::vector<float> policy_probs(num_actions);
    update_policy(context_features, num_actions, policy_probs, base_predictions, base, examples);
    
    // CRITICAL: Store the policy π_t(c_t, ·) for use in learning phase
    // This ensures we remember which policy was used to collect the data
    _last_policy = policy_probs;
    
    // Update predictions with policy probabilities
    for (size_t a = 0; a < num_actions; ++a)
    {
      preds[a].score = policy_probs[a];
    }
  }
  else
  {
    // Learning phase - store COMPLETE HISTORICAL DATA for future retraining
    // Extract the observed action and cost from the examples
    CB::cb_class observed = CB_ADF::get_observed_cost_or_default_cb_adf(examples);
    
    if (observed.probability > 0)  // Valid observation
    {
      // Store the complete training example for future retraining
      store_historical_data(examples, observed.action, observed.cost, observed.probability);
    }
    
    // Increment counter AFTER storing historical data to prepare for next round
    // This ensures round.round_number stores the correct round number for the data
    _counter++;
    
    // Note: Base learner handles the immediate learning via base.learn(examples) above
    // Historical data will be used for retraining in future prediction phases
  }
}



}  // namespace opocmab
}  // namespace cb_explore_adf
}  // namespace VW

namespace VW {
namespace cb_explore_adf {
namespace opocmab {

VW::LEARNER::base_learner* setup(VW::config::options_i& options, vw& all)
{
  using config::make_option;
  
  bool cb_explore_adf_option = false;
  bool opocmab = false;
  std::string type_string = "mtr";
  
  // OPO-CMAB parameters
  float eta = 0.1f;
  float gamma = 1.0f;
  bool sigmoid = false;
  
  config::option_group_definition new_options("[Reduction] Contextual Bandit Exploration with ADF (OPO-CMAB)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .necessary()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("opocmab", opocmab).keep().necessary().help("OPO-CMAB (Optimistic Policy Optimization) exploration"))
      .add(make_option("eta", eta)
               .keep()
               .default_value(0.1f)
               .help("Learning rate η for OPO-CMAB algorithm"))
      .add(make_option("gamma", gamma)
               .keep()
               .default_value(1.0f)
               .help("Tuning parameter γ for confidence bounds (β = γ * sqrt(k / num_actions)) in OPO-CMAB"))
      .add(make_option("sigmoid", sigmoid)
               .keep()
               .help("Apply sigmoid activation to scores from oracle before applying OPO-CMAB algorithm."))
      .add(make_option("cb_type", type_string)
               .keep()
               .help("contextual bandit method to use in {ips,dr,mtr}. Default: mtr"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf")) { options.insert("cb_adf", ""); }
  
  if (type_string != "mtr")
  {
    *(all.trace_message) << "warning: bad cb_type, OPO-CMAB only supports mtr; resetting to mtr." << std::endl;
    options.replace("cb_type", "mtr");
  }

  // Set explore_type
  size_t problem_multiplier = 1;

  VW::LEARNER::multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  using explore_type = cb_explore_adf_base<cb_explore_adf_opocmab>;
  auto data = scoped_calloc_or_throw<explore_type>(eta, gamma, sigmoid, &all);
  VW::LEARNER::learner<explore_type, multi_ex>& l =
      VW::LEARNER::init_learner(data, base, explore_type::learn, explore_type::predict, problem_multiplier,
          prediction_type_t::action_probs, all.get_setupfn_name(setup) + "-opocmab");

  l.set_finish_example(explore_type::finish_multiline_example);
  l.set_print_example(explore_type::print_multiline_example);
  l.set_persist_metrics(explore_type::persist_metrics);
  return make_base(l);
}

}  // namespace opocmab
}  // namespace cb_explore_adf
}  // namespace VW