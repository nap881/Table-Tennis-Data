# Table Tennis Win Rate Analysis & Match Predictor

This project analyzes one month of table tennis match data to understand what drives player win rates, and builds a machine learning model that predicts the outcome of a match between two players.

## Dataset

The data comes from a public Kaggle dataset: [One Month Table Tennis Dataset](https://www.kaggle.com/datasets/medaxone/one-month-table-tennis-dataset), which records more than 500 players' matches over the course of one month (`Setka.csv`).

The dataset contains 17 variables, including:

| Column | Description |
|---|---|
| `n` | Match number, sorted chronologically |
| `Date` | Date and time of the match |
| `Player1`, `Player2` | Player names |
| `Sets_P1`, `Sets_P2` | Sets won by each player (max 3) |
| `P(x)_G(y)` | Points won by player `x` in set `y` (`NA` if the set wasn't played) |
| `HomeWinner` | `1` if Player1 won, `0` if Player2 won |

A player is considered the winner of a match once they win 3 sets. Players with fewer than 10 total matches are excluded from most of the analysis so that win rate statistics are reasonably reliable.

## Data Analysis

### 1. Overall win rate distribution

For every player, win rate is computed as total wins divided by total matches played. Plotting this across all qualifying players produces a roughly normal distribution, with about 75% of players falling between a 30–70% win rate. This suggests most players cluster around an "intermediate" skill level — which lines up with the intuition that going from beginner to intermediate is relatively fast (~1 year of practice), while reaching an advanced level takes much longer (4–5+ years, plus coaching, talent, and equipment). The distribution is also slightly right-skewed, hinting that there may be more advanced players than beginners in the sample.

### 2. Win rate variance vs. overall win rate

Each player's **win rate variance** is defined as the difference between their highest and lowest single-day win rate (each day's win rate is computed from that day's wins ÷ matches). Players are sorted by this variance and split into three groups — Low, Medium, and High variance — and each group's daily win rates are plotted over time as well as summarized in tables (average total matches, overall win rate, highest/lowest daily win rate, win rate difference, and average days between matches).

Key findings:
- There is **no meaningful correlation** between a player's consistency (win rate variance) and their overall win rate. Low- and high-variance groups end up with very similar overall win rates, while the medium-variance group doesn't fall in between as might be expected.
- High-variance players tend to play **more matches over a longer span of time** than low-variance players, which naturally gives their win rate more days to fluctuate.
- This makes sense once you consider that win rate variance isn't purely a function of a player's own skill — it also depends heavily on the strength of the opponents faced on a given day. Two players with identical skill and consistency can end up with very different variance numbers just because of who they happened to play.

### 3. Opponent strength

To test whether a player's results are influenced by who they play, each player's **mean opponent win rate** is computed (the average win rate of everyone they've faced). Plotting a player's own win rate against their mean opponent win rate shows a weak but present positive correlation — opponent quality does have some effect on a player's results, meaning win rate alone isn't a perfectly clean measure of skill.

At the individual match level, a scatter plot compares Player 1's win rate at the time of the match (y-axis) against Player 2's win rate (x-axis), colored by who won. This shows that win rate difference is a strong predictor of outcome **only when the gap between the two players is large** — near the corners of the plot, the higher win-rate player wins almost every time. Toward the center, where players are more evenly matched, outcomes are far less predictable. The plot also shows visible grid lines at round percentages (25%, 33%, 50%, 67%, 75%), caused by players with exactly the minimum 10 matches, whose win rates are restricted to coarse increments.

## Match Outcome Prediction Model

### Goal

Rather than relying on career win rate alone, the model aims to produce a realistic win probability for any two players by combining **career performance**, **recent form**, and **opponent quality** into a single prediction.

### Data structure & leakage prevention

Matches are stored in long format (`Player1` vs. `Player2`, with `HomeWinner` indicating who won), so a helper function, `get_past_matches()`, standardizes results for a given player regardless of which column they appear in.

The most important design constraint was **avoiding data leakage**: every feature is computed using only matches that occurred *before* the match being predicted (`df['Date'] < match_date`), rather than using each player's full-season stats for every row. This is enforced through two core functions:

- `get_career_winrate(player, match_date, df)` — career win rate using only past matches
- `get_rolling_winrate(player, match_date, df, window=20)` — win rate over the last 20 matches before the given date, capturing recent form

This is more computationally expensive (rescanning a player's history for every match), which makes dataset construction take a few minutes, but it's necessary to get realistic, leakage-free evaluation results.

### Features

15 features are engineered per match, grouped into four categories:

- **Career performance**: `p1_win_rate`, `p2_win_rate`
- **Opponent quality**: `p1_opp_mean_win_rate`, `p2_opp_mean_win_rate`
- **Consistency & activity**: `p1_win_rate_variance`, `p2_win_rate_variance`, `p1_avg_days_between`, `p2_avg_days_between`
- **Recent form**: `p1_recent_wr`, `p2_recent_wr` (rolling 20-match window)

Plus differential features (`wr_diff`, `opp_wr_diff`, `variance_diff`, `avg_days_diff`, `recent_wr_diff`) that give the model pre-computed head-to-head comparisons between the two players.

### Train/test split

The data is split **chronologically** (80% earliest matches for training, 20% most recent for testing) rather than randomly, so the evaluation mirrors real deployment: training on the past, predicting the future.

### Model: Gradient Boosting

`GradientBoostingClassifier` (scikit-learn) was chosen because it:
- captures non-linear relationships and feature interactions without manual transforms,
- provides feature importances for interpretability, and
- is relatively insensitive to differing feature scales (win-rate percentages vs. day counts).

Hyperparameters were tuned with a 5-fold cross-validated `GridSearchCV` over:

```python
param_grid = {
    'n_estimators':  [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth':     [3, 4],
    'subsample':     [0.7, 0.8]
}
```

The best estimator from the search is used as the final model.

### Predicting a match

`predict_match(player1, player2)` computes the same 15 features for two given players (using the most recent date in the dataset as the cutoff), then calls `gbt.predict_proba()` to return a win probability for each player.

### Results

- **Test accuracy**: 58.2%
- **5-fold cross-validation accuracy**: 58.18% ± 0.39%

Table tennis outcomes are influenced by many factors not captured in this dataset — form on the day, stylistic matchups, psychology, injuries — so an accuracy in the high 50s means the model has learned a real, if noisy, signal from historical performance data. The small standard deviation across CV folds indicates the result is stable rather than a lucky train/test split.

### Future work

L1/L2 regularized models were tested but didn't improve results, likely because the feature set was already fairly compact. Promising next steps include adding head-to-head history, tournament importance, player rankings, match location, and trying other model families (XGBoost, LightGBM, ensembles).

## Repository Contents

- `ttdata.ipynb` — Jupyter notebook containing the full analysis pipeline: win rate calculation, win rate variance grouping, opponent-strength analysis, and the Gradient Boosting match predictor.
- `Setka.csv` — one-month table tennis match dataset (not included; download from Kaggle link above).

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

## Usage

1. Download `Setka.csv` from the Kaggle dataset linked above and place it in the project root.
2. Open and run `ttdata.ipynb` cell by cell to reproduce the analysis and train the model.
3. Once the model is trained, call `predict_match("Player Name 1", "Player Name 2")` to get a predicted winner and win probabilities for any two players in the dataset.
