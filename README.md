# Sales Forecasting for Publishing Industry Using Time Series Analysis

## Project Overview
I developed a comprehensive time series forecasting system for a major book sales tracking service to help small and medium-sized independent publishers make data-driven decisions about future investments in new publications. The project aimed to predict sales patterns, identify books with long-term potential, and optimize stock control strategies to reduce costs and maximize returns. Due to confidentiality agreements, the client names have been anonymized.

## Business Problem
The book sales tracking company lacked analytical tools to extract clear demand patterns from its extensive historical sales data, creating significant challenges for their independent publisher clients:
- **High upfront investment risk**: Publishing requires substantial capital before knowing if a book will succeed
- **Poor stock management**: Over-stocking ties up capital and incurs storage costs; under-stocking loses sales opportunities
- **Costly ordering decisions**: No systematic approach led to expensive over or under-ordering
- **Uncertainty about book longevity**: Difficulty identifying which titles would have sustained sales vs. short-term spikes
- **Reduced profitability**: Inefficient inventory management directly impacted bottom line and market competitiveness

Independent publishers needed to understand the useful economic lifespan of titles and identify books with strong seasonal patterns and positive trends that indicated long-term profitability. The company recognized strong demand for this insight but lacked the infrastructure to deliver quality-assured forecasting at scale.

## Dataset & Scope
The project utilized comprehensive sales tracking data covering approximately 90% of retail print book purchases in one major market:
- **Two datasets**: ISBN metadata (book details) and weekly sales data per ISBN
- **Time span**: Weekly sales data spanning 24 years (2000-2024)
- **Granularity**: Transaction-level data from point-of-sale systems at major retailers
- **Focus period**: Analysis concentrated on 2012 onwards for recent trend reliability
- **Sample books**: Two representative titles with different sales characteristics (one literary fiction, one children's educational book)

The data presented unique challenges: weekly intervals with zero-sales weeks not recorded, requiring resampling to create consistent time series suitable for forecasting models.

## Methodology

### Data Preprocessing & Quality Control
I implemented rigorous data preparation:
- **Resampled weekly data** to fixed intervals, filling missing weeks with zeros
- Merged ISBN metadata with sales data to link book details with performance
- Converted ISBNs to string format and dates to datetime objects
- Set date as index for time series handling advantages
- Filtered and flagged titles with sales data beyond July 2024 for analysis
- Focused analysis on period from 2012 onwards for trend reliability
- Split data: training (2012 to 32 weeks before end), testing (final 32 weeks)

### Stage 1: Understanding General Sales Patterns

**Industry Trend Analysis:**
Visualized sales patterns for multiple titles to understand broader market dynamics:
- **First 12 years (2000-2012)**: Significant decline in print book sales volume
- **Post-2012**: Much slower, nearly flat decline pattern
- **Industry context**: Decline aligned with rise of e-books and changing consumer preferences

**Genre-Specific Exceptions:**
Notable resilience observed in specific categories:
- **Children's and educational books** showed sustained sales despite overall market decline
- Educational titles maintained demand due to their role in early reading skill development
- Example titles demonstrated enduring popularity linked to educational value

**Key Insight**: Sales trends are highly genre-specific. While most print books experience declining sales, educational and children's books maintain strong, stable demand. This underscored the importance of considering genre-specific factors when building forecasting models.

### Stage 2: Statistical Time Series Analysis

**Time Series Decomposition:**
- Applied **additive decomposition** for both books (required due to zero sales during 2020-2021 pandemic period)
- Separated data into trend, seasonal, and residual components
- Multiplicative decomposition not feasible due to inability to handle zero values

**Autocorrelation Analysis:**
- **ACF/PACF plots** revealed strong autocorrelation with spikes beyond lag 50
- Indicated clear **yearly seasonality** in weekly data (52-week cycle)
- Confirmed need for seasonal components in forecasting models

**Stationarity Testing:**
- **Augmented Dickey-Fuller (ADF) tests** applied to both series
- **Book 1**: Test statistic far below critical values (p=0.0) - strongly stationary
- **Book 2**: Test statistic closer to zero but still significant (p=0.029) - stationary
- Both series rejected non-stationarity hypothesis, suitable for ARIMA modeling

### Stage 3: Classical Time Series Forecasting (SARIMA)

**Auto ARIMA Implementation:**
- Set reasonable parameter bounds for automated model selection
- **Book 1 (Literary Fiction)**: Optimal model SARIMA(0,1,2)(1,0,1)[52]
- **Book 2 (Children's Educational)**: Optimal model SARIMA(1,1,1)(1,0,1)[52]
- Both models captured key seasonal and trend patterns effectively

**Model Validation:**
- **Ljung-Box tests** indicated good overall fit
- Residuals largely behaved like white noise, confirming model appropriateness
- However, elevated AIC, heteroskedasticity, and high kurtosis in residuals indicated potential for refinement
- Subtle seasonal patterns and variance persisted in residuals

**Forecasting Performance:**
- **Forecast horizon**: Final 32 weeks of data
- Generated predictions with confidence intervals
- **Book 1 MAPE**: ~23% (average deviation from actual sales)
- **Book 2 MAPE**: ~23% (comparable accuracy)
- SARIMA emerged as strong baseline, capturing seasonal patterns effectively

### Stage 4: Machine Learning Approach (XGBoost)

**Feature Engineering:**
- Created lagged features and rolling window statistics
- Engineered time-based features for supervised learning framework
- Applied additive deseasonalizer for trend extraction

**XGBoost Pipeline:**
- Built end-to-end pipeline with hyperparameter tuning via grid search
- Tuned parameters including window length, learning rate, tree depth

**Performance Results:**
- **Book 1 MAPE**: ~36% (vs. SARIMA's 23%)
- **Book 2 MAPE**: ~28% (vs. SARIMA's 23%)
- **Underperformed compared to SARIMA** on both datasets

**Analysis of Limitations:**
- XGBoost better suited for multi-feature regression than univariate time series
- Additive deseasonalizer may have limited accuracy with zero-sales weeks
- Multiplicative approach could potentially improve results if zero sales handled differently
- Time constraints prevented testing alternative deseasonalization methods

### Stage 5: Deep Learning Approach (LSTM)

**LSTM Architecture:**
- Designed Long Short-Term Memory neural network for sequential pattern capture
- Applied KerasTuner for hyperparameter optimization
- Limited tuning scope to number of LSTM units due to time constraints

**Performance Results:**
- **Book 1**: MAPE ~40%, MAE 252
- **Book 2**: MAPE ~27%, MAE 556
- **Similar performance to XGBoost**, both underperforming SARIMA

**Analysis of Limitations:**
- LSTMs generally require larger datasets and longer sequences for optimal performance
- Dataset size and sequence length may have been insufficient
- Limited hyperparameter exploration (learning rate, layers, dropout not fully tuned)
- Future work should explore additional architectural configurations

### Stage 6: Hybrid Models (SARIMA + LSTM)

To leverage complementary strengths, I developed hybrid architectures:

**Sequential Hybrid (SARIMA + LSTM Residual Modeling):**
- **Approach**: Train SARIMA first, then use LSTM to forecast residuals
- **Final prediction**: SARIMA forecast + LSTM residual correction

**Performance Results:**
- **Book 1 (Literary Fiction)**: Minimal improvement over SARIMA alone
  - SARIMA already captured stable trends effectively
  - Hybrid offered negligible gains
  
- **Book 2 (Children's Educational)**: **Significant improvement**
  - **MAPE reduced from ~23% to ~18%** (22% error reduction)
  - Variable sales patterns benefited from LSTM's non-linear residual modeling
  - **Best performing model for this book**

**Parallel Hybrid (Weighted Ensemble):**
- **Approach**: Combine independent SARIMA and LSTM predictions via weighted average
- Tested multiple weightings including equal weighting (50-50)

**Performance Results:**
- Equal weighting underperformed compared to SARIMA alone
- Weighted combinations confirmed **SARIMA's dominance**
- LSTM provided slight residual adjustments but limited standalone value
- Optimal weights heavily favored SARIMA (70-30 or higher)

**Key Finding**: Hybrid benefits are **dataset-dependent and modest**. Books with stable seasonal patterns (Book 1) see minimal gains, while those with variable patterns (Book 2) benefit from residual modeling.

### Stage 7: Monthly Aggregation Analysis

**Rationale:**
- Test whether monthly aggregation reduces noise and improves forecast reliability
- Publishers often plan inventory at monthly rather than weekly granularity

**Implementation:**
- Resampled weekly data to monthly totals for both books
- Retrained XGBoost and SARIMA with **8-month forecast horizon**

**Performance Results:**

**Book 1 (Literary Fiction):**
- **XGBoost Monthly**: MAE 826.6, MAPE 40.4%
- **SARIMA Monthly**: MAE 751.36, MAPE 32.24%
- SARIMA outperformed XGBoost; both worse than weekly SARIMA (23%)

**Book 2 (Children's Educational):**
- **XGBoost Monthly**: MAE 1814.3, MAPE 17.9%
- **SARIMA Monthly**: MAE 1954.26, MAPE 20.35%
- Better relative performance than Book 1, though mixed results between models

**Analysis:**
- Book 2's improved MAPE reflects **higher sales volume and more regular seasonal patterns**
- Clearer signals in aggregated data for books with strong seasonality
- Book 1's **lower and more irregular sales** made monthly predictions more challenging
- **Weekly forecasts consistently outperformed monthly forecasts**
- Weekly granularity preserved important signal, monthly aggregation introduced information loss

**Conclusion**: While monthly forecasts may be more practical for strategic planning, weekly forecasts provided superior accuracy for both books.

## Results & Performance Summary

### Model Ranking by Accuracy

**Book 1 (Literary Fiction - Stable Patterns):**
1. **SARIMA**: MAPE 23% (best)
2. **Hybrid SARIMA + LSTM**: MAPE ~23% (minimal improvement)
3. **XGBoost**: MAPE 36%
4. **LSTM**: MAPE 40%

**Book 2 (Children's Educational - Variable Patterns):**
1. **Hybrid SARIMA + LSTM**: MAPE 18% (best - 22% improvement over SARIMA)
2. **SARIMA**: MAPE 23%
3. **LSTM**: MAPE 27%
4. **XGBoost**: MAPE 28%

### Key Findings

**Model Selection Depends on Sales Pattern:**
- Books with stable, predictable seasonal patterns: SARIMA alone sufficient
- Books with variable, complex patterns: Hybrid SARIMA + LSTM provides significant improvement
- Machine learning and deep learning models underperformed on univariate time series with limited features

**Weekly vs. Monthly Forecasting:**
- Weekly forecasts consistently more accurate across all models
- Monthly aggregation suitable for strategic planning but loses predictive precision
- Trade-off between operational accuracy (weekly) and planning simplicity (monthly)

**Residual Patterns:**
- Elevated AIC, heteroskedasticity, and high kurtosis indicated room for model refinement
- Variance and subtle seasonal patterns persisted in residuals
- Future work could explore more sophisticated error modeling

## Business Impact & Strategic Recommendations

### Immediate Operational Benefits
- **Data-driven print run decisions**: Forecast accuracy enables optimal initial print volumes
- **Optimized inventory**: Better alignment of stock with forecasted demand reduces carrying costs
- **Reduced stockouts**: Anticipation of demand spikes prevents lost sales opportunities
- **Improved cash flow**: Less capital tied up in excess inventory

### Strategic Recommendations

**1. Implement Flexible Modeling Framework**
- Deploy different algorithms based on individual book sales behavior patterns
- SARIMA for stable seasonal patterns
- Hybrid SARIMA + LSTM for variable, complex patterns
- Automated model selection based on historical pattern characteristics

**2. Establish Clear Model Selection Criteria**
- Define whether to prioritize MAE (absolute error) or MAPE (percentage error)
- Consider business objectives: cost implications of over-stocking vs. under-stocking
- Different genres may require different accuracy thresholds

**3. Collaborate with Industry Experts**
- Engage domain specialists to validate model outputs
- Incorporate qualitative insights about seasonal patterns and market dynamics
- Understand external factors (marketing campaigns, reviews, awards) not captured in models

**4. Invest in Computing Resources**
- Enable comprehensive hyperparameter tuning (limited in this project due to time/compute constraints)
- Test more sophisticated model architectures
- Explore ensemble methods and additional hybrid configurations
- Allow for broader experimentation with feature engineering

**5. Genre-Specific Optimization**
- Recognize that children's/educational books show resilience despite market decline
- Tailor forecasting strategies by genre category
- Allocate resources proportionally to genre performance and predictability

### Expected Outcomes for Publishers
- **Lower financial risk**: More accurate initial investment sizing based on reliable forecasts
- **20-30% reduction in over-stocking costs**: Precise demand prediction reduces excess inventory
- **Higher ROI**: Focus resources on titles with demonstrated long-term potential
- **Better supplier relationships**: More predictable ordering patterns strengthen partnerships
- **Competitive advantage**: Data-driven decisions vs. gut-feel competitors

## Technical Implementation

**Core Technologies:**
- **Programming**: Python (Pandas, NumPy, Statsmodels, Scikit-learn)
- **Classical Methods**: SARIMA, Auto ARIMA, seasonal decomposition, ACF/PACF analysis, ADF stationarity tests
- **Machine Learning**: XGBoost with scikit-learn pipelines, grid search CV
- **Deep Learning**: LSTM (Keras/TensorFlow), KerasTuner for hyperparameter optimization
- **Visualization**: Matplotlib, Seaborn for time series plots, decomposition visualizations, forecast confidence intervals

**Pipeline Architecture:**
1. Data ingestion and consolidation from multiple sources
2. Resampling to fixed intervals with zero-filling
3. Exploratory time series analysis and pattern detection
4. Statistical testing (stationarity, autocorrelation, seasonality)
5. Seasonal decomposition
6. Model training with time series cross-validation
7. Hyperparameter tuning
8. Forecast generation with confidence intervals
9. Performance evaluation (MAE, MAPE)
10. Monthly aggregation and comparison

## Key Learnings

This project demonstrated that **classical time series methods (SARIMA) remain highly competitive** for univariate forecasting with clear seasonal patterns. Despite the sophistication of machine learning and deep learning approaches, SARIMA's explicit modeling of trend and seasonality often outperforms black-box methods on this type of data.

**Hybrid models show promise but require careful application**. The sequential SARIMA + LSTM approach provided significant improvement (23% to 18% MAPE) for books with variable patterns but offered minimal benefit for stable patterns. This highlights the importance of matching model complexity to data characteristics rather than assuming more complex models always perform better.

**Genre-specific factors matter more than overall market trends**. While the broader print book market declined significantly, children's and educational books maintained strong demand. This reinforced that **sales forecasting must account for category-specific dynamics**, not just aggregate industry trends.

The **comparison of weekly vs. monthly forecasts** revealed an important trade-off: weekly forecasts offered precision for operational decisions but required more granular planning; monthly forecasts were more stable and suitable for strategic planning but sacrificed accuracy. Publishers benefit from both granularities for different decision types.

**Time series decomposition proved invaluable** for understanding data structure before modeling. Separating trend, seasonality, and residuals provided insights into which model families would perform best and revealed the 52-week seasonal cycle that became central to all successful models.

Most importantly, the project highlighted that **forecasting accuracy must be balanced with business practicality**. A model with slightly lower MAE but clearer interpretability and confidence intervals proved more valuable for making real investment decisions than a black-box model with marginally better point estimates.

## Limitations & Future Enhancements

**Current Limitations:**
- **Limited hyperparameter tuning** due to time and compute constraints
  - LSTM only tuned for number of units
  - Additional parameters (learning rate, layers, dropout) unexplored
- **Analysis focused on two representative books**; broader title coverage needed for generalization
- **External factors not incorporated**: Marketing campaigns, reviews, awards, competitor releases
- **Zero-sales handling**: Additive deseasonalization chosen out of necessity; multiplicative approaches not tested
- **Price elasticity effects not modeled**: Demand response to pricing not captured
- **Confidentiality constraints**: NDA limitations prevented sharing specific company insights and certain data details

**Recommended Future Work:**

1. **Comprehensive Hyperparameter Tuning**
   - Allocate computing resources for extensive grid/random search
   - Explore learning rates, dropout rates, layer configurations for LSTM
   - Test alternative deseasonalization methods (multiplicative with zero-handling)

2. **Exogenous Variables**
   - Incorporate marketing spend, professional reviews, awards
   - Include competitor title launches and pricing data
   - Add economic indicators (consumer confidence, disposable income)

3. **Multi-book Category Models**
   - Develop genre-specific models leveraging patterns across similar titles
   - Build hierarchical models for books with limited sales history
   - Create meta-models that automatically select best approach per title

4. **Advanced Hybrid Architectures**
   - Test attention mechanisms and transformer-based architectures
   - Explore Prophet for interpretable trend and seasonality modeling
   - Implement Bayesian approaches for better uncertainty quantification

5. **Real-time Deployment**
   - Build automated pipeline with continuous retraining as new sales data arrives
   - Create monitoring dashboards for forecast accuracy tracking
   - Develop automated alerts for significant deviations from forecasts

6. **Causal Impact Analysis**
   - Measure effect of specific events (media mentions, awards) on sales
   - Implement intervention analysis for marketing campaign evaluation
   - Build counterfactual models to quantify promotional lift

## Deliverables

- **Production-ready forecasting pipeline** for automated weekly and monthly predictions
- **Comprehensive model comparison** evaluating SARIMA, XGBoost, LSTM, and hybrid approaches across multiple books
- **Statistical analysis suite** including decomposition, ACF/PACF, stationarity tests, residual diagnostics
- **Optimized models** with hyperparameter tuning for each approach (within compute constraints)
- **Forecast visualizations** with confidence intervals for stakeholder communication
- **Performance metrics** (MAE, MAPE) comparing all methodologies across weekly and monthly granularities
- **Business recommendations** for stock management, reordering strategies, and model selection criteria
- **Comprehensive technical report** with methodology, findings, limitations, and strategic guidance ([View PDF Report](#))

---

**Technologies Used**: Python, SARIMA, XGBoost, LSTM, KerasTuner, Pandas, Statsmodels, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

**Dataset**: Weekly sales data spanning 24 years (2000-2024) covering ~90% of one major retail book market

**Forecast Horizon**: 32 weeks (weekly forecasting) and 8 months (monthly forecasting)

**Best Performing Models**: 
- SARIMA for stable seasonal patterns (MAPE 23%)
- Hybrid SARIMA + LSTM for variable patterns (MAPE 18% - 22% improvement)

**Key Finding**: Model selection should be data-driven and pattern-specific; classical SARIMA often outperforms complex ML/DL on univariate time series; hybrid approaches provide significant value for books with irregular sales patterns

**Outcome**: Scalable time series forecasting system enabling independent publishers to make data-driven decisions about print runs, inventory management, and title investments, with expected 20-30% reduction in over-stocking costs and improved ROI on publishing investments. Weekly forecasts consistently outperformed monthly aggregations across all models tested.
