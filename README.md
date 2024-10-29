Electricity Market Bid Data
This dataset provides hourly bid data from the Italian electricity market, retrieved from the MercatoElettrico FTP server. 
Each XML file represents a single day's market activity, capturing information from market bids (including both offers and demands), though only demand bids (BID types) are retained in this processed dataset. 
The dataset focuses specifically on certain market zones, filtered for time-series analysis, and is provided as a pickled DataFrame file.

Data Overview
Each XML file contains hourly records for a single day, represented by the following fields:

Day: Date in YYYYMMDD format.
Hour: The hour associated with each bid (0–23).
Quantity: The amount related to the bid in MWh.
Price: The bid price in Euros per MWh.
Market (Mercato): Constant field equal to MGP (Mercato del Giorno Prima).
Market Zones (ZonaMercato): The full list of market zones participating in the auction.
Bid Type (Tipo): Indicates the type of bid, either OFF (offer) or BID (demand). Only demand bids are retained in the final dataset.
Each instance in the dataset corresponds to an individual bid, not a daily or hourly aggregate. This level of granularity allows for a detailed examination of bidding patterns and pricing.

Preprocessing Steps
FTP Access: XML files are downloaded daily over a defined date range.
Data Parsing: Each XML file is processed, and relevant bid data is parsed into a DataFrame.
Filtering:
Only demand bids (Type "BID")across all participating market zones (ZonaMercato) are retained.
The dataset is then reduced to three main columns: timestamp, Quantity, and Price.
Timestamp Conversion: Each record’s date and hour fields are combined into a single timestamp column, facilitating easier indexing and analysis.

NONPARAMETRIC STATISTICS PROJECT OBJECTIVE:
Tentative Goals: Functional Analysis of Hourly Bidding Patterns in the Electricity Market
Objective: To treat daily bid profiles as functional data, analyzing continuous variations in prices and quantities over hourly intervals to understand evolving trends.
Pssible Research Questions:
1-How do daily profiles of bid prices and quantities evolve over time, and can we detect significant functional differences across days or zones?
2-Using survival analysis, can we estimate the likelihood that prices will remain within a certain standard deviation over time?
Possible Methods from class
Functional Data Analysis (FDA): Model daily bidding profiles as functional curves, applying techniques like Functional Principal Component Analysis (FPCA) to capture dominant patterns and cycles.
Functional ANOVA (FANOVA): Test for significant differences in functional profiles across days, zones, or other factors.
Survival Analysis: Examine the persistence of price levels within set thresholds, using techniques like Kaplan-Meier estimates or Cox proportional hazards models for "survival" within defined price limits.
Nonparametric Hypothesis Tests: Apply tests such as the Mann-Whitney U-test or Kruskal-Wallis test to assess the statistical significance of differences in bid prices and quantities across zones or times.
