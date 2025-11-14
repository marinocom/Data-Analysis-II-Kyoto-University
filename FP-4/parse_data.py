"""
parse_data.py

This module parses the flight data files for the FP-4 assignment.
Extracts relevant columns and filters data for Philadelphia International Airport (PHL).
"""

import pandas as pd


# -----------------------------
# Load Flight Data Function
# -----------------------------
def load_flight_data(filepath):
    """
    Load flight data from CSV file with appropriate data types.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing flight data
        
    Returns:
    --------
    pandas.DataFrame
        Raw flight data with proper data types
    """
    # Fix mixed type warning: cancellation/diversion columns are sparse categorical data
    df = pd.read_csv(filepath,
                     dtype={
                         'CancellationCode': str,
                         'Div1TailNum': str,
                         'Div2Airport': str,
                         'Div2TailNum': str,
                         'Div3Airport': str,
                         'Div3TailNum': str
                     })
    return df


# -----------------------------
# Filter by Origin Airport Function
# -----------------------------
def filter_by_origin(df, origin_airport='PHL'):
    """
    Filter flights by origin airport.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw flight data
    origin_airport : str, optional
        Airport code to filter by (default: 'PHL')
        
    Returns:
    --------
    pandas.DataFrame
        Filtered flight data for specified origin airport
    """
    df_filtered = df[df['Origin'] == origin_airport].copy()
    return df_filtered


# -----------------------------
# Extract Dependent and Independent Variables Columns Function
# -----------------------------
def extract_relevant_columns(df):
    """
    Extract relevant columns for analysis.
    
    The dependent and independent variables (DVs and IVs) are:
    
    DVs:
    - ArrDelay (Arrival Delay in minutes)
    
    IVs:
    - DepDelay (Departure delay in minutes)
    - DayOfWeek
    - CRSDepTime (Computer Reservation System Departure Time)
    - Reporting_Airline (Airline Carrier Code)
    - Dest
    - Distance
    - Month
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Filtered flight data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with only relevant columns
    """
    columns = ['ArrDelay', 'DepDelay', 'DayOfWeek', 'CRSDepTime', 
               'Reporting_Airline', 'Dest', 'Distance', 'Month']
    
    df_relevant = df[columns].copy()
    return df_relevant


# -----------------------------
# Rename Columns Function
# -----------------------------
def rename_columns(df):
    """
    Rename columns to simpler variable names.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with original column names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with renamed columns
    """
    df_renamed = df.rename(columns={
        'ArrDelay': 'arrival_delay',
        'DepDelay': 'departure_delay',
        'DayOfWeek': 'day_of_week',
        'CRSDepTime': 'scheduled_dep_time',
        'Reporting_Airline': 'airline',
        'Dest': 'destination',
        'Distance': 'distance',
        'Month': 'month'
    })
    return df_renamed


# -----------------------------
# Main Parse Data Function
# -----------------------------
def parse_flight_data(filepath, origin_airport='PHL'):
    """
    Main function to parse flight data.
    
    This function:
    1. Loads the raw flight data
    2. Filters by origin airport (default: PHL)
    3. Extracts relevant columns
    4. Renames columns to simpler names
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing flight data
    origin_airport : str, optional
        Airport code to filter by (default: 'PHL')
        
    Returns:
    --------
    pandas.DataFrame
        Processed flight data ready for analysis
    """
    # Load data
    df_raw = load_flight_data(filepath)
    
    # Filter by origin airport
    df_filtered = filter_by_origin(df_raw, origin_airport)
    
    # Extract relevant columns
    df_relevant = extract_relevant_columns(df_filtered)
    
    # Rename columns
    df_final = rename_columns(df_relevant)
    
    return df_final