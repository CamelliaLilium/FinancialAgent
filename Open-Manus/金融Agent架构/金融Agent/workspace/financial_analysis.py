text = 'However, the volatility of these and other commodity prices is such that possible future increases in these prices remain a risk to the inflation outlook.'

# Determine if the statement is hawkish, dovish, or neutral
# A hawkish statement typically indicates a tightening of monetary policy, which is usually associated with raising interest rates or reducing money supply to combat inflation.
# A dovish statement suggests an easing of monetary policy, often associated with lowering interest rates or increasing money supply to stimulate economic growth.
# A neutral statement is even-handed and does not clearly indicate a shift in policy.

# Analyze the text for keywords related to inflation and monetary policy
keywords_hawkish = ['tightening', 'increase', 'raise', 'higher', 'inflation', 'risk', 'volatility', 'prices']
keywords_dovish = ['easing', 'lower', 'decrease', 'reduce', 'stimulate', 'growth', 'support']

# Check for hawkish keywords
hawkish_keywords = [keyword for keyword in keywords_hawkish if keyword in text]
# Check for dovish keywords
 dovish_keywords = [keyword for keyword in keywords_dovish if keyword in text]

# Determine the stance based on the presence of keywords
if len(hawkish_keywords) > len(dovish_keywords):
    answer = 'HAWKISH'
elif len(dovish_keywords) > len(hawkish_keywords):
    answer = 'DOVISH'
else:
    answer = 'NEUTRAL'

answer