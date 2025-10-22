# Task 2 Results Summary

This document summarizes the expected deliverables and outputs from Task 2.

## Expected Outputs

### 1. Trained Models
After running the notebook, you will have three trained models saved in:
- `outputs/roberta/` - RoBERTa model checkpoints
- `outputs/deberta/` - DeBERTa model checkpoints  
- `outputs/modernbert/` - ModernBERT model checkpoints

### 2. Evaluation Results

#### CSV File: `outputs/model_comparison.csv`
Contains F1-scores for all three models:
```
Model,F1-Score (Macro),F1-Score (Weighted)
RoBERTa,0.9xxx,0.9xxx
DeBERTa,0.9xxx,0.9xxx
ModernBERT,0.9xxx,0.9xxx
```

#### Visualizations:
- `outputs/model_f1_comparison.png` - Bar chart comparing F1-scores
- `outputs/confusion_matrices.png` - Confusion matrices for all three models
- `outputs/rpp_category_distribution.png` - Category distribution in RPP predictions
- `outputs/model_vs_llm_comparison.png` - (Bonus) Comparison with LLM if available

### 3. RPP News Predictions

#### CSV File: `outputs/rpp_predictions.csv`
Contains predictions from all models for RPP articles:
```
title,roberta_pred,roberta_category,deberta_pred,deberta_category,modernbert_pred,modernbert_category
"Article Title",0,World,0,World,1,Sports
...
```

### 4. Bonus: LLM Classifications

#### JSON File: `data/rpp_classified.json` (optional)
Contains LLM classifications as ground truth:
```json
{
  "model_used": "gpt-4o-mini",
  "num_articles": 50,
  "llm_classifications": [0, 1, 2, ...],
  "detailed_results": [...],
  "category_mapping": {
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Science/Technology"
  }
}
```

## Typical Results

Based on the AG News dataset, you can expect:

### Model Performance (AG News Test Set)
- **F1-Score Range**: 0.88 - 0.95 (all models)
- **Training Time**: 1-3 hours (depending on hardware)
- **Best Model**: Typically ModernBERT or DeBERTa show slight improvements

### Category-Specific Performance
- **Sports**: Usually highest F1 (0.92-0.98) - clear linguistic patterns
- **Sci/Tech**: High F1 (0.90-0.96) - distinct terminology
- **Business**: Good F1 (0.88-0.94) - some overlap with World
- **World**: Good F1 (0.88-0.94) - most diverse category

### Model Agreement on RPP News
- **All models agree**: 60-80% of articles
- **Sports articles**: Highest agreement (85-95%)
- **Mixed-topic articles**: Lower agreement (40-60%)

## Discussion Points

### Why differences exist between models:

1. **Architecture**:
   - RoBERTa: Dynamic masking, larger batches
   - DeBERTa: Disentangled attention, better context
   - ModernBERT: Recent optimizations, efficient training

2. **Pretraining**:
   - Different corpus sizes and sources
   - Different pretraining objectives
   - Different tokenization strategies

3. **Capacity**:
   - Parameter counts vary (125M-183M)
   - Depth and width of networks
   - Attention mechanism complexity

### Domain Shift (AG News → RPP News):

1. **Language**: English (AG) vs Spanish (RPP)
   - Models trained on English may not perform well
   - Consider multilingual models for production

2. **Geographic Focus**: International vs Peru/Latin America
   - Local references may be misclassified
   - Cultural context matters

3. **Topic Distribution**: 
   - AG News: Balanced distribution
   - RPP News: May skew toward certain categories
   - Temporal effects (news from specific dates)

## Recommendations

1. **For English news classification**: Use the best-performing model (likely DeBERTa or ModernBERT)

2. **For Spanish news classification**: 
   - Fine-tune multilingual models (mBERT, XLM-R)
   - Collect Spanish news data for training
   - Use data augmentation

3. **For production deployment**:
   - Consider model size vs accuracy tradeoff
   - Implement confidence thresholds
   - Use ensemble for critical applications
   - Monitor performance over time

4. **For ambiguous articles**:
   - Show multiple category predictions with probabilities
   - Allow manual review and correction
   - Use feedback to improve model

## Next Steps

After completing Task 2:

1. ✅ Review model performance metrics
2. ✅ Analyze confusion matrices for insights
3. ✅ Examine examples where models disagree
4. ✅ Consider ensemble approaches
5. ✅ Document findings and interpretations
6. ✅ (Bonus) Compare with LLM classifications

## Troubleshooting

### Common Issues:

1. **Out of Memory**: 
   - Reduce batch_size in notebook
   - Use smaller models
   - Enable gradient checkpointing

2. **Slow Training**:
   - Reduce num_epochs
   - Use GPU if available
   - Reduce dataset size for testing

3. **Poor Performance**:
   - Check data preprocessing
   - Verify labels are correct
   - Increase training epochs
   - Adjust learning rate

4. **Model Download Fails**:
   - Check internet connection
   - Use cached models if available
   - Try alternative model sources
