"""
classical.py

This module conducts insightful hypothesis testing analyses for flight delay data.
Goes beyond simple comparisons to explore nuanced patterns and interactions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from IPython.display import display, Markdown
import seaborn as sns

# -----------------------------
# Utility Functions
# -----------------------------
def display_title(s, pref='Figure', num=1):
    """Display formatted title for analysis sections."""
    s = f'<p><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></p>'
    display(Markdown(s))


def display_result(test_name, statistic, p_value, interpretation):
    """Display formatted test results."""
    result = f"""
**{test_name}**
- Test Statistic: {statistic:.4f}
- p-value: {p_value:.4f}
- Interpretation: {interpretation}
"""
    display(Markdown(result))


# -----------------------------
# Analysis 1: Cascading Delay Effect by Flight Distance
# -----------------------------
def analyze_cascading_delay_by_distance(df, short_dist_threshold=500, long_dist_threshold=1000):
    """
    INSIGHTFUL ANALYSIS: Tests whether the relationship between departure and 
    arrival delays differs for short vs. long flights.
    
    Hypothesis: Short flights have less opportunity to "make up time" in the air,
    so departure delays cascade more directly to arrival delays. Long flights
    can recover during cruise, showing weaker correlation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Flight data with delay and distance information
    short_dist_threshold : int
        Upper bound for short flights (miles)
    long_dist_threshold : int
        Lower bound for long flights (miles)
        
    Returns:
    --------
    dict : Results including correlation coefficients and statistical tests
    """
    display_title('Cascading Delay Effect by Flight Distance', num=2)
    
    # Remove NaN values
    df_clean = df.dropna(subset=['departure_delay', 'arrival_delay', 'distance'])
    
    # Separate into short and long flights
    short_flights = df_clean[df_clean['distance'] <= short_dist_threshold]
    long_flights = df_clean[df_clean['distance'] >= long_dist_threshold]
    
    # Calculate correlations
    r_short = np.corrcoef(short_flights['departure_delay'], 
                          short_flights['arrival_delay'])[0, 1]
    r_long = np.corrcoef(long_flights['departure_delay'], 
                         long_flights['arrival_delay'])[0, 1]
    
    # Fisher z-transformation to compare correlations
    z_short = np.arctanh(r_short)
    z_long = np.arctanh(r_long)
    
    n_short = len(short_flights)
    n_long = len(long_flights)
    
    se_diff = np.sqrt(1/(n_short-3) + 1/(n_long-3))
    z_stat = (z_short - z_long) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, data, label, r in zip(axs, 
                                   [short_flights, long_flights],
                                   ['Short Flights (≤500 mi)', 'Long Flights (≥1000 mi)'],
                                   [r_short, r_long]):
        ax.scatter(data['departure_delay'], data['arrival_delay'], 
                  alpha=0.3, s=10)
        
        # Add regression line
        x = data['departure_delay'].values
        y = data['arrival_delay'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)
        
        ax.set_xlabel('Departure Delay (min)', fontsize=11)
        ax.set_ylabel('Arrival Delay (min)', fontsize=11)
        ax.set_title(f'{label}\nr = {r:.3f}', fontsize=12)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results
    interpretation = f"Short flights show {'significantly stronger' if p_value < 0.05 else 'similar'} correlation (r={r_short:.3f}) compared to long flights (r={r_long:.3f}). "
    if p_value < 0.05:
        interpretation += "This suggests pilots on longer flights can make up time during cruise."
    else:
        interpretation += "The cascading delay effect appears similar across flight distances."
    
    display_result('Fisher Z-test for Correlation Difference', z_stat, p_value, interpretation)
    
    return {
        'r_short': r_short,
        'r_long': r_long,
        'z_statistic': z_stat,
        'p_value': p_value,
        'n_short': n_short,
        'n_long': n_long
    }


# -----------------------------
# Analysis 2: Peak Hour Delays - Time-of-Day Effects
# -----------------------------
def analyze_peak_hour_congestion(df, morning_hours=(6, 9), evening_hours=(17, 20)):
    """
    INSIGHTFUL ANALYSIS: Compares arrival delays during peak congestion hours
    vs. off-peak hours, controlling for departure delays.
    
    Hypothesis: Flights arriving during peak hours (morning/evening rush) 
    experience additional delays beyond their departure delays due to 
    airspace and runway congestion.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Flight data with delay and scheduled departure time
    morning_hours : tuple
        Morning peak hours range
    evening_hours : tuple
        Evening peak hours range
        
    Returns:
    --------
    dict : Results including mean differences and t-test statistics
    """
    display_title('Peak Hour Congestion Effect on Delays', num=2)
    
    # Calculate scheduled arrival hour (approximate)
    df_clean = df.dropna(subset=['scheduled_dep_time', 'arrival_delay', 
                                  'departure_delay', 'distance']).copy()
    
    # Estimate flight duration in hours (rough approximation: distance/500 mph)
    flight_duration_hours = df_clean['distance'] / 500.0
    dep_hour = df_clean['scheduled_dep_time'] // 100
    arr_hour = (dep_hour + flight_duration_hours) % 24
    
    # Create "excess delay" metric: how much worse arrival delay is than departure delay
    df_clean.loc[:, 'excess_delay'] = df_clean['arrival_delay'] - df_clean['departure_delay']
    
    # Identify peak vs off-peak arrivals
    peak_mask = ((arr_hour >= morning_hours[0]) & (arr_hour < morning_hours[1])) | \
                ((arr_hour >= evening_hours[0]) & (arr_hour < evening_hours[1]))
    
    peak_delays = df_clean[peak_mask]['excess_delay']
    offpeak_delays = df_clean[~peak_mask]['excess_delay']
    
    # Remove extreme outliers for cleaner comparison (>3 std devs)
    peak_delays = peak_delays[np.abs(stats.zscore(peak_delays)) < 3]
    offpeak_delays = offpeak_delays[np.abs(stats.zscore(offpeak_delays)) < 3]
    
    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(peak_delays, offpeak_delays, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((peak_delays.std()**2 + offpeak_delays.std()**2) / 2)
    cohens_d = (peak_delays.mean() - offpeak_delays.mean()) / pooled_std
    
    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    data_to_plot = [offpeak_delays, peak_delays]
    bp = axs[0].boxplot(data_to_plot, labels=['Off-Peak', 'Peak Hours'],
                        patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'salmon']):
        patch.set_facecolor(color)
    axs[0].set_ylabel('Excess Delay (min)\n(Arrival - Departure)', fontsize=11)
    axs[0].set_title('Delay Distribution by Arrival Time', fontsize=12)
    axs[0].grid(alpha=0.3, axis='y')
    axs[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Histogram
    axs[1].hist(offpeak_delays, bins=50, alpha=0.6, label='Off-Peak', color='lightblue', density=True)
    axs[1].hist(peak_delays, bins=50, alpha=0.6, label='Peak Hours', color='salmon', density=True)
    axs[1].set_xlabel('Excess Delay (min)', fontsize=11)
    axs[1].set_ylabel('Density', fontsize=11)
    axs[1].set_title('Distribution Comparison', fontsize=12)
    axs[1].legend()
    axs[1].grid(alpha=0.3, axis='y')
    axs[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Results
    interpretation = f"Peak hour arrivals show mean excess delay of {peak_delays.mean():.2f} min vs. {offpeak_delays.mean():.2f} min for off-peak. "
    if p_value < 0.05:
        interpretation += f"This difference is statistically significant (Cohen's d = {cohens_d:.3f}), suggesting congestion adds delays beyond departure issues."
    else:
        interpretation += "This difference is not statistically significant."
    
    display_result("Welch's t-test for Peak Hour Effect", t_stat, p_value, interpretation)
    
    return {
        'peak_mean': peak_delays.mean(),
        'offpeak_mean': offpeak_delays.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'n_peak': len(peak_delays),
        'n_offpeak': len(offpeak_delays)
    }


# -----------------------------
# Analysis 3: Weekly Pattern Stratification
# -----------------------------
def analyze_weekday_vs_weekend_by_airline(df, top_n_airlines=3):
    """
    INSIGHTFUL ANALYSIS: Tests whether delay patterns differ between weekdays
    and weekends, stratified by airline to control for carrier effects.
    
    Hypothesis: Weekend flights experience different delay patterns than weekday
    flights due to passenger mix (leisure vs. business) and operational patterns.
    This effect may vary by airline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Flight data with day_of_week, airline, and delay information
    top_n_airlines : int
        Number of top airlines to analyze
        
    Returns:
    --------
    dict : Results for each airline including t-test statistics
    """
    display_title('Weekday vs Weekend Delays Stratified by Airline', num=3)
    
    df_clean = df.dropna(subset=['arrival_delay', 'day_of_week', 'airline']).copy()
    
    # Create weekday/weekend indicator (1-5: weekday, 6-7: weekend)
    df_clean.loc[:, 'is_weekend'] = df_clean['day_of_week'].isin([6, 7])
    
    # Get top airlines by flight count
    top_airlines = df_clean['airline'].value_counts().head(top_n_airlines).index
    
    results = {}
    fig, axs = plt.subplots(1, top_n_airlines, figsize=(15, 4))
    if top_n_airlines == 1:
        axs = [axs]
    
    for idx, airline in enumerate(top_airlines):
        airline_data = df_clean[df_clean['airline'] == airline]
        
        weekday_delays = airline_data[~airline_data['is_weekend']]['arrival_delay']
        weekend_delays = airline_data[airline_data['is_weekend']]['arrival_delay']
        
        # Remove extreme outliers
        weekday_delays = weekday_delays[np.abs(stats.zscore(weekday_delays)) < 3]
        weekend_delays = weekend_delays[np.abs(stats.zscore(weekend_delays)) < 3]
        
        # T-test
        t_stat, p_value = stats.ttest_ind(weekday_delays, weekend_delays, equal_var=False)
        
        # Visualization
        ax = axs[idx]
        data_to_plot = [weekday_delays, weekend_delays]
        bp = ax.boxplot(data_to_plot, labels=['Weekday', 'Weekend'],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightyellow']):
            patch.set_facecolor(color)
        ax.set_ylabel('Arrival Delay (min)', fontsize=10)
        ax.set_title(f'Airline: {airline}\np = {p_value:.4f}', fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        results[airline] = {
            'weekday_mean': weekday_delays.mean(),
            'weekend_mean': weekend_delays.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'n_weekday': len(weekday_delays),
            'n_weekend': len(weekend_delays)
        }
    
    plt.tight_layout()
    plt.show()
    
    # Display results
    for airline, result in results.items():
        interpretation = f"Airline {airline}: Weekday mean = {result['weekday_mean']:.2f} min, Weekend mean = {result['weekend_mean']:.2f} min. "
        if result['p_value'] < 0.05:
            interpretation += "Significant difference suggests operational or passenger mix effects."
        else:
            interpretation += "No significant difference detected."
        display_result(f"t-test for Airline {airline}", result['t_statistic'], 
                      result['p_value'], interpretation)
    
    return results


# -----------------------------
# Analysis 4: Seasonal Effect on Delay Recovery
# -----------------------------
def analyze_seasonal_delay_recovery(df, winter_months=[12, 1, 2], summer_months=[6, 7, 8]):
    """
    INSIGHTFUL ANALYSIS: Tests whether the ability to recover from departure
    delays (make up time in flight) differs between winter and summer months.
    
    Hypothesis: Winter weather creates more persistent delays that are harder
    to recover from, while summer conditions allow better in-flight recovery.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Flight data with month and delay information
    winter_months : list
        Months classified as winter
    summer_months : list
        Months classified as summer
        
    Returns:
    --------
    dict : Results including regression coefficients and comparison tests
    """
    display_title('Seasonal Effects on Delay Recovery', num=3)
    
    df_clean = df.dropna(subset=['arrival_delay', 'departure_delay', 'month'])
    
    # Focus on delayed departures only (where recovery is possible)
    df_delayed = df_clean[df_clean['departure_delay'] > 0].copy()
    
    # Calculate recovery rate: (dep_delay - arr_delay) / dep_delay
    # Positive = made up time, Negative = got worse
    df_delayed['recovery_rate'] = ((df_delayed['departure_delay'] - df_delayed['arrival_delay']) / 
                                   df_delayed['departure_delay'])
    
    # Remove outliers
    df_delayed = df_delayed[np.abs(stats.zscore(df_delayed['recovery_rate'])) < 3]
    
    # Separate seasons
    winter_data = df_delayed[df_delayed['month'].isin(winter_months)]
    summer_data = df_delayed[df_delayed['month'].isin(summer_months)]
    
    # Compare recovery rates
    t_stat, p_value = stats.ttest_ind(winter_data['recovery_rate'], 
                                       summer_data['recovery_rate'], 
                                       equal_var=False)
    
    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot comparison
    data_to_plot = [winter_data['recovery_rate'], summer_data['recovery_rate']]
    bp = axs[0].boxplot(data_to_plot, labels=['Winter', 'Summer'],
                        patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'orange']):
        patch.set_facecolor(color)
    axs[0].set_ylabel('Recovery Rate\n(Proportion of Delay Recovered)', fontsize=11)
    axs[0].set_title('Delay Recovery by Season', fontsize=12)
    axs[0].grid(alpha=0.3, axis='y')
    axs[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Recovery')
    axs[0].legend()
    
    # Scatter plot with regression lines
    for data, label, color in [(winter_data, 'Winter', 'blue'),
                                (summer_data, 'Summer', 'orange')]:
        axs[1].scatter(data['departure_delay'], data['arrival_delay'], 
                      alpha=0.2, s=10, label=label, color=color)
        
        # Regression line
        x = data['departure_delay'].values
        y = data['arrival_delay'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        axs[1].plot(x_line, p(x_line), color=color, linewidth=2)
    
    # Add y=x line (no recovery)
    max_val = max(df_delayed['departure_delay'].max(), df_delayed['arrival_delay'].max())
    axs[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No Recovery')
    
    axs[1].set_xlabel('Departure Delay (min)', fontsize=11)
    axs[1].set_ylabel('Arrival Delay (min)', fontsize=11)
    axs[1].set_title('Departure vs Arrival Delay by Season', fontsize=12)
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results
    interpretation = f"Winter recovery rate: {winter_data['recovery_rate'].mean():.3f}, Summer: {summer_data['recovery_rate'].mean():.3f}. "
    if p_value < 0.05:
        if summer_data['recovery_rate'].mean() > winter_data['recovery_rate'].mean():
            interpretation += "Summer flights show significantly better delay recovery, likely due to favorable weather conditions."
        else:
            interpretation += "Winter flights show significantly better delay recovery, which is unexpected and warrants further investigation."
    else:
        interpretation += "No significant seasonal difference in delay recovery detected."
    
    display_result("Welch's t-test for Seasonal Recovery Difference", 
                  t_stat, p_value, interpretation)
    
    return {
        'winter_mean_recovery': winter_data['recovery_rate'].mean(),
        'summer_mean_recovery': summer_data['recovery_rate'].mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'n_winter': len(winter_data),
        'n_summer': len(summer_data)
    }



# -----------------------------
# Main Analysis Function
# -----------------------------
def run_all_analyses(df, selection=[1,2,3,4]):

    # Run all insightful analyses on the flight data.
    results = {}
    
    if 1 in selection:
        # Analysis 1: Distance-dependent cascading delays
        results['cascading_delay'] = analyze_cascading_delay_by_distance(df)
        print("\n\n")

    if 2 in selection:
        # Analysis 2: Peak hour congestion effects
        results['peak_hour'] = analyze_peak_hour_congestion(df)
        print("\n\n")

    if 3 in selection:
    # Analysis 3: Weekday vs weekend by airline
        results['weekday_weekend'] = analyze_weekday_vs_weekend_by_airline(df)
        print("\n\n")
    
    if 4 in selection:
        # Analysis 4: Seasonal delay recovery
        results['seasonal_recovery'] = analyze_seasonal_delay_recovery(df)
        print("\n\n")