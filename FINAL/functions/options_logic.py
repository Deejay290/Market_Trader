import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
import yfinance as yf
from functions import calculations

def calculate_option_pnl(option_type, strike_price, premium, underlying_prices):
    """Calculates Profit/Loss for a single option leg at expiration."""
    pnl = []
    if option_type == 'call':
        for price in underlying_prices:
            profit_loss = max(0, price - strike_price) - premium 
            pnl.append(profit_loss)
    elif option_type == 'put':
        for price in underlying_prices:
            profit_loss = max(0, strike_price - price) - premium 
            pnl.append(profit_loss)
    return pnl

def calculate_breakeven(option_type, strike_price, premium):
    """Calculate breakeven price for option."""
    if option_type == 'call':
        return strike_price + premium
    elif option_type == 'put':
        return strike_price - premium
    return None

def calculate_intrinsic_value(option_type, strike_price, current_price):
    """Calculate intrinsic value of option."""
    if option_type == 'call':
        return max(0, current_price - strike_price)
    elif option_type == 'put':
        return max(0, strike_price - current_price)
    return 0

def calculate_time_value(last_price, intrinsic_value):
    """Calculate time value (extrinsic value) of option."""
    return max(0, last_price - intrinsic_value)

def rank_options_logic(df_original, option_type, current_price_for_rank, dte_for_rank):
    """
    Enhanced ranking system with stricter quality filters and better scoring.
    """
    df = df_original.copy()
    
    # Ensure delta exists
    if 'delta' not in df.columns:
        df['delta'] = np.nan 

    # CRITICAL FILTERS - Remove obviously bad options
    required_cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:  
        return pd.DataFrame()  

    # Filter 1: Valid prices (avoid division by zero and nonsense data)
    df = df[(df['lastPrice'] > 0.01) & (df['bid'] > 0) & (df['ask'] > 0)]
    
    # Filter 2: Reasonable IV (remove extreme outliers)
    df = df[(df['impliedVolatility'] > 0) & (df['impliedVolatility'] < 5.0)]
    
    # Filter 3: Minimum liquidity threshold (adjustable based on stock)
    min_volume = 10  # Minimum daily volume
    min_oi = 50  # Minimum open interest
    df = df[(df['volume'] >= min_volume) & (df['openInterest'] >= min_oi)]
    
    # Filter 4: Reasonable bid-ask spread (max 20% of mid-price)
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['spread_pct'] = ((df['ask'] - df['bid']) / (df['mid_price'] + 1e-9)) * 100
    df = df[df['spread_pct'] <= 20]
    
    if df.empty:
        return pd.DataFrame()

    # Calculate POP with enhanced error handling
    df.loc[:, 'POP'] = df.apply(
        lambda row: calculations.calculate_pop(
            option_type, 
            row['strike'], 
            row['lastPrice'], 
            current_price_for_rank, 
            row['impliedVolatility'], 
            dte_for_rank
        ), 
        axis=1
    )
    df['POP'].fillna(0, inplace=True)
    
    # Calculate additional metrics
    df.loc[:, 'intrinsic_value'] = df.apply(
        lambda row: calculate_intrinsic_value(option_type, row['strike'], current_price_for_rank),
        axis=1
    )
    df.loc[:, 'time_value'] = df.apply(
        lambda row: calculate_time_value(row['lastPrice'], row['intrinsic_value']),
        axis=1
    )
    df.loc[:, 'breakeven'] = df.apply(
        lambda row: calculate_breakeven(option_type, row['strike'], row['lastPrice']),
        axis=1
    )
    
    # Distance to breakeven as percentage
    df.loc[:, 'breakeven_distance_pct'] = abs(
        (df['breakeven'] - current_price_for_rank) / (current_price_for_rank + 1e-9) * 100
    )
    
    # --- ENHANCED SCORING SYSTEM ---
    
    # 1. POP Score (40%) - Higher weight for probability
    df.loc[:, 'pop_score'] = df['POP']
    
    # 2. Liquidity Score (25%) - Critical for execution
    max_volume = df['volume'].max()
    min_volume_val = df['volume'].min()
    max_oi = df['openInterest'].max()
    min_oi_val = df['openInterest'].min()
    
    df.loc[:, 'norm_volume'] = (df['volume'] - min_volume_val) / (max_volume - min_volume_val + 1e-9) if (max_volume - min_volume_val) != 0 else 0.5
    df.loc[:, 'norm_oi'] = (df['openInterest'] - min_oi_val) / (max_oi - min_oi_val + 1e-9) if (max_oi - min_oi_val) != 0 else 0.5
    
    # Spread score (tighter spread = better)
    df.loc[:, 'spread_score'] = 1 - (df['spread_pct'] / 20)  # Normalized to 0-1
    df['spread_score'] = df['spread_score'].clip(0, 1)
    
    df.loc[:, 'liquidity_score'] = (
        df['norm_volume'] * 0.4 + 
        df['norm_oi'] * 0.4 + 
        df['spread_score'] * 0.2
    )
    
    # 3. Value Score (20%) - Risk/reward consideration
    # Prefer options with good time value relative to price (not overpaying)
    df.loc[:, 'time_value_ratio'] = df['time_value'] / (df['lastPrice'] + 1e-9)
    df.loc[:, 'time_value_ratio'] = df['time_value_ratio'].clip(0, 1)
    
    # Breakeven achievability score (closer breakeven = higher score for reasonable moves)
    # Ideal breakeven is 3-8% away for calls, 3-8% for puts
    ideal_breakeven_distance = 5.0  # 5% is ideal
    df.loc[:, 'breakeven_score'] = 1 - abs(df['breakeven_distance_pct'] - ideal_breakeven_distance) / 10
    df['breakeven_score'] = df['breakeven_score'].clip(0, 1)
    
    df.loc[:, 'value_score'] = (
        df['time_value_ratio'] * 0.4 +
        df['breakeven_score'] * 0.6
    )
    
    # 4. Delta Score (15%) - Probability proxy
    df["delta"] = df["delta"].fillna(0.5)
    
    if option_type == 'call':
        # For calls, prefer delta 0.3-0.7 (balance of leverage and probability)
        df.loc[:, "delta_score"] = df["delta"].apply(
            lambda x: 1 - abs(x - 0.5) / 0.5 if 0.2 <= x <= 0.8 else 0.2
        )
    else:  # put
        # For puts, prefer absolute delta 0.3-0.7
        df.loc[:, "delta_score"] = df["delta"].apply(
            lambda x: 1 - abs(abs(x) - 0.5) / 0.5 if 0.2 <= abs(x) <= 0.8 else 0.2
        )
    
    # Final weighted score
    df.loc[:, 'Overall_Score'] = (
        df['pop_score'] * 0.40 +          
        df['liquidity_score'] * 0.25 +    
        df['value_score'] * 0.20 +        
        df['delta_score'] * 0.15           
    )
    
    # Add quality tier classification
    df.loc[:, 'Quality_Tier'] = pd.cut(
        df['Overall_Score'],
        bins=[0, 0.4, 0.6, 0.75, 1.0],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )

    return df.sort_values("Overall_Score", ascending=False)

def analyze_single_option_details(row, current_price, dte, option_type):
    """
    ENHANCED comprehensive analysis with clear buy/no-buy recommendation.
    """
    price = row["lastPrice"]
    iv = row["impliedVolatility"]
    volume, oi = row["volume"], row["openInterest"]
    delta = row.get("delta", 0.0) if pd.notnull(row.get("delta", 0.0)) else 0.0
    strike = row['strike']
    bid = row['bid']
    ask = row['ask']
    
    # Calculate all metrics
    pop = calculations.calculate_pop(option_type, strike, price, current_price, iv, dte)
    intrinsic = calculate_intrinsic_value(option_type, strike, current_price)
    time_value = calculate_time_value(price, intrinsic)
    breakeven = calculate_breakeven(option_type, strike, price)
    
    # Distance to breakeven
    breakeven_distance = abs(breakeven - current_price)
    breakeven_pct = (breakeven_distance / (current_price + 1e-9)) * 100
    
    # Spread analysis
    spread = ask - bid
    spread_pct = (spread / (price + 1e-9)) * 100 if price > 0 else 0
    
    # Risk metrics
    max_loss = price  # For long options
    
    # --- SCORING SYSTEM (0-100) ---
    score = 0
    max_score = 100
    reasons = []
    red_flags = []
    
    # 1. Probability of Profit (25 points)
    if pd.isna(pop) or pop == 0:
        reasons.append("âš ï¸ **POP N/A** - Cannot calculate probability (check IV/DTE)")
        pop_score = 0
    elif pop >= 0.55:
        pop_score = 25
        reasons.append(f"âœ… **High POP: {pop*100:.1f}%** - Strong probability of profit")
    elif pop >= 0.45:
        pop_score = 20
        reasons.append(f"ðŸ‘ **Moderate POP: {pop*100:.1f}%** - Reasonable probability")
    elif pop >= 0.35:
        pop_score = 15
        reasons.append(f"â„¹ï¸ **Fair POP: {pop*100:.1f}%** - Below average probability")
    else:
        pop_score = 5
        red_flags.append(f"ðŸš© **Low POP: {pop*100:.1f}%** - Poor probability of profit")
    
    score += pop_score
    
    # 2. Liquidity (25 points)
    liquidity_score = 0
    
    # Volume check
    if volume >= 500:
        liquidity_score += 10
        reasons.append(f"âœ… **Excellent Volume: {volume:,.0f}** - Very liquid")
    elif volume >= 100:
        liquidity_score += 7
        reasons.append(f"ðŸ‘ **Good Volume: {volume:,.0f}** - Adequate liquidity")
    elif volume >= 50:
        liquidity_score += 4
        reasons.append(f"â„¹ï¸ **Moderate Volume: {volume:,.0f}** - Acceptable but watch slippage")
    else:
        red_flags.append(f"ðŸš© **Low Volume: {volume:,.0f}** - May have execution issues")
    
    # Open Interest check
    if oi >= 1000:
        liquidity_score += 10
        reasons.append(f"âœ… **High Open Interest: {oi:,.0f}** - Strong market interest")
    elif oi >= 500:
        liquidity_score += 7
        reasons.append(f"ðŸ‘ **Good Open Interest: {oi:,.0f}** - Decent market depth")
    elif oi >= 100:
        liquidity_score += 4
        reasons.append(f"â„¹ï¸ **Moderate Open Interest: {oi:,.0f}** - Limited market depth")
    else:
        red_flags.append(f"ðŸš© **Low Open Interest: {oi:,.0f}** - Very illiquid")
    
    # Spread check
    if spread_pct <= 5:
        liquidity_score += 5
        reasons.append(f"âœ… **Tight Spread: {spread_pct:.1f}%** - Low transaction cost")
    elif spread_pct <= 10:
        liquidity_score += 3
        reasons.append(f"ðŸ‘ **Acceptable Spread: {spread_pct:.1f}%** - Reasonable cost")
    elif spread_pct <= 15:
        liquidity_score += 1
        reasons.append(f"â„¹ï¸ **Wide Spread: {spread_pct:.1f}%** - High transaction cost")
    else:
        red_flags.append(f"ðŸš© **Very Wide Spread: {spread_pct:.1f}%** - Expensive to trade")
    
    score += liquidity_score
    
    # 3. Implied Volatility (15 points)
    iv_score = 0
    if 0.15 <= iv <= 0.60:
        iv_score = 15
        reasons.append(f"âœ… **Healthy IV: {iv:.1%}** - Fairly priced premium")
    elif 0.60 < iv <= 1.0:
        iv_score = 10
        reasons.append(f"â„¹ï¸ **Elevated IV: {iv:.1%}** - Premium is expensive but big moves expected")
    elif iv < 0.15:
        iv_score = 8
        red_flags.append(f"ðŸš© **Very Low IV: {iv:.1%}** - Cheap but limited movement expected")
    else:
        iv_score = 5
        red_flags.append(f"ðŸš© **Extreme IV: {iv:.1%}** - Very expensive premium")
    
    score += iv_score
    
    # 4. Moneyness/Delta (20 points)
    delta_score = 0
    if option_type == 'call':
        if 0.60 <= delta <= 0.80:
            delta_score = 20
            reasons.append(f"âœ… **Strong Delta: {delta:.2f}** - High probability ITM, good leverage")
        elif 0.40 <= delta < 0.60:
            delta_score = 18
            reasons.append(f"âœ… **Balanced Delta: {delta:.2f}** - ATM sweet spot")
        elif 0.25 <= delta < 0.40:
            delta_score = 14
            reasons.append(f"ðŸ‘ **Moderate Delta: {delta:.2f}** - Decent leverage, lower probability")
        elif 0.15 <= delta < 0.25:
            delta_score = 8
            reasons.append(f"â„¹ï¸ **Low Delta: {delta:.2f}** - OTM, needs significant move")
        else:
            delta_score = 3
            red_flags.append(f"ðŸš© **Very Low Delta: {delta:.2f}** - Far OTM, lottery ticket")
    else:  # put
        abs_delta = abs(delta)
        if 0.60 <= abs_delta <= 0.80:
            delta_score = 20
            reasons.append(f"âœ… **Strong Delta: {delta:.2f}** - High probability ITM")
        elif 0.40 <= abs_delta < 0.60:
            delta_score = 18
            reasons.append(f"âœ… **Balanced Delta: {delta:.2f}** - ATM sweet spot")
        elif 0.25 <= abs_delta < 0.40:
            delta_score = 14
            reasons.append(f"ðŸ‘ **Moderate Delta: {delta:.2f}** - Decent leverage")
        elif 0.15 <= abs_delta < 0.25:
            delta_score = 8
            reasons.append(f"â„¹ï¸ **Low Delta: {delta:.2f}** - OTM, needs significant move")
        else:
            delta_score = 3
            red_flags.append(f"ðŸš© **Very Low Delta: {delta:.2f}** - Far OTM")
    
    score += delta_score
    
    # 5. Time Value & Breakeven (15 points)
    value_score = 0
    time_value_pct = (time_value / (price + 1e-9)) * 100 if price > 0 else 0
    
    if breakeven_pct <= 3:
        value_score += 8
        reasons.append(f"âœ… **Close Breakeven: {breakeven_pct:.1f}%** - Needs only small move")
    elif breakeven_pct <= 6:
        value_score += 6
        reasons.append(f"ðŸ‘ **Reasonable Breakeven: {breakeven_pct:.1f}%** - Moderate move needed")
    elif breakeven_pct <= 10:
        value_score += 3
        reasons.append(f"â„¹ï¸ **Distant Breakeven: {breakeven_pct:.1f}%** - Significant move needed")
    else:
        red_flags.append(f"ðŸš© **Very Distant Breakeven: {breakeven_pct:.1f}%** - Large move required")
    
    if 30 <= time_value_pct <= 70:
        value_score += 7
        reasons.append(f"âœ… **Balanced Time Value: {time_value_pct:.0f}%** - Fair pricing")
    elif time_value_pct < 30:
        value_score += 4
        reasons.append(f"â„¹ï¸ **Low Time Value: {time_value_pct:.0f}%** - Mostly intrinsic")
    else:
        value_score += 2
        red_flags.append(f"ðŸš© **High Time Value: {time_value_pct:.0f}%** - Paying mostly for time")
    
    score += value_score
    
    # --- FINAL RECOMMENDATION ---
    recommendation = ""
    confidence = ""
    
    if score >= 80 and len(red_flags) == 0:
        recommendation = "ðŸš€ STRONG BUY"
        confidence = "High Confidence - Excellent setup across all metrics"
    elif score >= 70 and len(red_flags) <= 1:
        recommendation = "âœ… BUY"
        confidence = "Good Confidence - Strong overall profile"
    elif score >= 60:
        recommendation = "ðŸ‘ CONSIDER"
        confidence = "Moderate Confidence - Decent setup but monitor closely"
    elif score >= 50:
        recommendation = "âš ï¸ CAUTION"
        confidence = "Low Confidence - Significant concerns present"
    else:
        recommendation = "ðŸ›‘ AVOID"
        confidence = "Not Recommended - Too many risk factors"
    
    # Risk/Reward calculation
    # Estimate target as 2x premium (100% return) and stop as 50% loss
    target_profit = price * 1.0  # 100% gain
    stop_loss = price * 0.5  # 50% loss
    rr_ratio = target_profit / (stop_loss + 1e-9) if stop_loss > 0 else np.nan
    
    return (
        recommendation, 
        confidence,
        score,
        max_score,
        reasons, 
        red_flags,
        max_loss, 
        target_profit, 
        rr_ratio, 
        pop,
        breakeven,
        breakeven_pct
    )