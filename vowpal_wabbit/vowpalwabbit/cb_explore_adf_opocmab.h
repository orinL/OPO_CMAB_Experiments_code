// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "cb_explore_adf_common.h"
#include "reductions_fwd.h"

#include <vector>
#include <map>
#include <memory> // Required for std::unique_ptr

namespace VW
{
namespace cb_explore_adf
{
namespace opocmab
{

/*
OPO-CMAB implementation with efficient weight snapshot management.
Uses weight snapshots instead of retraining for theoretically correct past predictions.
*/
struct cb_explore_adf_opocmab
{
private:
  size_t _counter;
  float _learning_rate;
  float _gamma;
  bool _sigmoid;
  
  // Store the policy from the last prediction for use in learning
  std::vector<float> _last_policy;
  
  // Historical training data for theoretically correct approach
  struct HistoricalRound {
    // Memory-safe storage for VW examples with proper RAII cleanup
    std::vector<std::unique_ptr<example, void(*)(example*)>> examples;
    uint32_t chosen_action;
    float observed_cost;
    float probability;
    size_t round_number;
    std::vector<float> policy_used;
    
    // Constructor
    HistoricalRound() = default;
    
    // Move constructor and assignment (no copying to avoid double-deletion)
    HistoricalRound(HistoricalRound&& other) noexcept = default;
    HistoricalRound& operator=(HistoricalRound&& other) noexcept = default;
    
    // Disable copy constructor and assignment to prevent memory issues
    HistoricalRound(const HistoricalRound&) = delete;
    HistoricalRound& operator=(const HistoricalRound&) = delete;
    
    // Destructor is automatic via unique_ptr with custom deleter
    ~HistoricalRound() = default;
    
    // Helper method to add examples safely
    void add_example(const example* src_example);
    
    // Helper method to get examples for prediction (const access)
    const std::vector<std::unique_ptr<example, void(*)(example*)>>& get_examples() const { return examples; }
  };
  std::vector<HistoricalRound> _historical_data;
  
  // Cache for predictions to avoid recomputation
  std::map<std::pair<size_t, size_t>, std::vector<float>> _prediction_cache;
  
  // Weight snapshot management for efficient past predictions
  std::vector<std::vector<float>> _weight_snapshots;
  std::vector<float> _temp_weight_backup;
  vw* _all;

public:
  cb_explore_adf_opocmab(float eta, float gamma, bool sigmoid, vw* all);
  ~cb_explore_adf_opocmab() = default;

  // Main prediction and learning interface
  template <bool is_learn>
  void predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples);

  void predict(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<false>(base, examples); }
  void learn(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<true>(base, examples); }

private:
  // Core algorithm functions
  void update_policy(const std::vector<float>& context_features, size_t num_actions, 
                     std::vector<float>& policy_probs, const std::vector<float>& base_predictions, 
                     VW::LEARNER::multi_learner& base, const multi_ex& current_examples);

  float compute_beta(size_t k, float gamma, size_t num_actions);

  // Past prediction computation using weight snapshots
  float compute_past_prediction_on_current_context(VW::LEARNER::multi_learner& base, 
                                                   const multi_ex& current_examples, 
                                                   size_t k, size_t action);

  // Weight snapshot management functions
  void save_current_weights();
  void restore_weights_from_snapshot(size_t snapshot_index);
  void save_weight_snapshot_after_round();
  void restore_current_weights();

  // Utility functions
  std::vector<float> extract_context_features(const multi_ex& examples);
  size_t compute_context_hash(const std::vector<float>& context_features);
  void clear_prediction_cache();
  void store_historical_data(const multi_ex& examples, uint32_t chosen_action, float cost, float prob);
  
  // Helper function for training data reconstruction
  multi_ex reconstruct_training_example(const multi_ex& original_examples, 
                                         uint32_t chosen_action, 
                                         float observed_cost, 
                                         float probability);
};

}  // namespace opocmab
}  // namespace cb_explore_adf

namespace cb_explore_adf
{
namespace opocmab
{
VW::LEARNER::base_learner* setup(VW::config::options_i& options, vw& all);
}  // namespace opocmab
}  // namespace cb_explore_adf
}  // namespace VW