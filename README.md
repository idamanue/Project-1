# Introduction
Traditional banks often deny loan applications from borrowers with bad or no credit. Such scheme is being disrupted by a Fintech domain known as *peer to peer (P2P) lending*. You are likely still eligible to get a loan with a fair interest rate – even if you have bad credit.
Unlike traditional banks, P2P loans are funded by individuals (or groups of individuals) through online platforms with very low overhead. This is the key difference which makes P2P personal loans for bad or no credit an attractive option. When traditional banks have no choice but to decline a loan application, P2P lenders can offer a financial product with reasonable, fixed interest rates. An example of such platform is *Lending Tree*.

Another option for people with bad or no credit is through a government's loan guarantee program. Such program involves one lending institution and multiple borrowers that are approved in the program. The purpose is to encourage a lender to extend credit to areas that are underserved by financial institutions. An example of such program is the USAID's Development Credit Authority (DCA). The DCA works with investors, local financial institutions, and development organizations to design and deliver investment alternatives that unlock financing for U.S. Government priorities.

The purpose of this project is to help people who need money (borrowers) and those who have money (investors) in their decision making.
This project has two parts:
In the first part, we use loan data from LendingClub.com to analyze the risks associated with a borrower with bad or no credit. The goal is to identify, from the data, the characteristics of borrowers that are more likely to default on their loans. We call them ***At Risk*** borrowers.
In the second part, we use data from DCA to provide an outlook of the U.S. Government priorities both in terms of geographical regions and activity sectors.

   
and the second part, kape_part2.ipynb, analyzes aid data from a borrower standpoint.

# Part 1: "At Risk" Borrowers Characteristics

The file ***kape_part1.ipynb*** provides the code for this analysis. 

## Dataset: Loan Data
The dataset is from 2007 to 2010 and we'll be trying to classify and predict whether or not the borrower paid back their loan in full.
After going through the data and understanding the available columns, we selected the following features of the borrower

* loan_status: Status of the loan (fully paid, late, defaulted, etc.)
* public_record = "tax_liens" + "pub_rec_bankruptcies": The borrower’s number of derogatory public records (bankruptcy filings, tax liens, or judgments)
* fico = last_fico_range_low: the FICO credit score of the borrower
* revol_util: The borrower’s revolving line utilization rate (the amount of the credit line used relative to total credit available)
* revol_bal: The borrower’s revolving balance (amount unpaid at the end of the credit card billing cycle).
* home_ownership: The ownership status of the applicant's residence (RENT, MORTGAGE, OWN, etc.)
* annual_inc: The natural log of the self-reported annual income of the borrower. 
    - because of the size, the natural log of the income will be used in the analysis: log_annual_inc = np.log(annual_inc)
* hardship_flag: Y or N or nan
* delinq_2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years
* dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income)
* mths_since_recent_inq: Months since the last delinquency.
* emp_length: Number of years in the job, rounded down. If longer than 10 years, then this is represented by the value 10
* inq_last_6mths: The borrower’s number of inquiries by creditors in the last 6 months
* zip_code: 3-digit zipcode prefix for the borrower.

The below columns are also included, as features of the loan itself.

* loan_amount: The amount of the loan the applicant received.
    - because of the size, the natural log of the loan amount will be used in the analysis: log_loan_amount = np.log(loan_amount)
* term: The number of months of the loan the applicant received.
* int_rate: Interest rate of the loan the applicant received.
* purpose: The purpose of the loan (takes values “creditcard”, “debtconsolidation”, “educational”, “majorpurchase”, “smallbusiness”, and “all_other”)

## Methodology
### Purpose and Data Source
The purpose of this part of the project is twofolds:  
(1) To determine those features, other that the credit score, that affect the interest rate on a loan  
(2) to predict whether or not a borrower is at risk of defaulting on a loan based on some characteristics.   

We focus on microloans.
We define Microloans as loans that do not exceed 100,000 USD. 
The dataset used, Loan_data.csv, was downloaded from Kragle and represent loans from LendingTree, which provides a very large, open dataset on  people who received loans through their platform.

### Data Cleansing
* Print all the columns to scan through and identify features of interest
* Managing columns: combining and renaming some columns as described in the data description
* Extracting the columns and rows of interest. For example, the data include different type of loans, indicated by the *application_type*. We focus on individual loans; so we will extract those records with *application_type == "Individual"*
* Drop the null values

### Checking and adjusting Data Structure
We observe that the *term*, *emp_length*, and *zip_code* are objects. They should have been numeric.
A quick inspection of each field reveals the following formats:
1. **term**: number + "months
2. **emp_length**: num + "years" with length over 10 years provided in the format "10+" + "years"
3. **zip_code**: 3 digit prefix + "xx"

The following refinement was applied:
* Only 2 terms are reported; either 36 months or 60 months. It will be maintained as a categorical variable
* emp_length: We will remove any non-digits character after the number; meaning 10+ years will be considered 10 years.
* zip_code: we will remove the xx

### Variable Correlation
With the aid of a correlation matrix and heat map, we identify the correlations among the features of the borrower. This allows us to further reduce the number of variables needed. For example, it was found that there was a very strong negative correlation between the number of delinquencies the past 2 years and the number of months since last delinquency. This allows us to ignore the mths_since_last_delinq from further analysis; knowing that if the variable *delinq_2yrs* ends up being significant, then the risks associated with higher number of delinquincies in the past 2 years will also be associated with smaller number of months since last delinquency.
![Correlation Heatmap](part1_heatmap.png)

### Distributions
We then look at how the cummulative loans or the median interest rate is distributed against the loan features. For example, it was determined that though the majority (more than 50%) of loans are received by home owners followed by renters, home ownership doesn't seem to affect the interest rate.

### How do the features of a borrower affect the loan interest rate?
We then plot the median rate against each of the remaining features of the borrowers.



### Predicting Default
A borrower's loan status indicates whether or not he/she has defaulted, fully paid, late, etc.  
We'll assume two levels:
* **Fully Paid** if the loan has been paid in full.
* **At Risk** otherwise    
An overlay plot of "At Risk" and "Fully Paid" plots provides the features that would most likely put a borrower at risk of defaulting on a loan.


# Part 2: USAID's DCA Loans Distributions

The file ***kape_part2.ipynb*** provides the code for this analysis. 

## Dataset: USAID DCA Data

* Amount (USD): Amount of the loan 
* Disbursement Date: The date the loan was disbursed 
* Business Sector: sector of activity
* City/Town: city or town depending on the region
* Latitude: A float indicating the geographical coordinate
* Longitude: A float indicating the geographical coordinate 
* Is Woman Owned? Whether or not the activity or small business or project being financed is owned by a woman
* Gender of owner: A float indicating the gender of the small business or project 
* Is First Time Borrower? A Boolean indicating whether or not the borrower is getting the loan for the first time 
* Business Size: A string indicating the size of the business or project


## Methodology
### Purpose and Data Source

In this part of the project, we focus on loans provided by DCA to developing countries around the World; how are they distributed and which sectors do they target the most.
The dataset used, [Development_Credit_Authority_DCA.csv](https://data.usaid.gov/Economic-Growth/Development-Credit-Authority-DCA-Data-Set-Loan-Tra/dt8c-833c), was downloaded from USAID.gov website and represent the list of all private loans made under USAID's DCA since from 1999 to 2019.

### Data cleansing and Reduction
The same process as in part 1 is used for data cleansing and reduction.
We then look at:
* Distribution of DCA loans by country
* Distribution of DCA loans by sectors
* Distribution of DCA loans by individual recipients
* A map plot to identify the predominant sector of activities and geographical locations in the World.

# Conclusion
From part 1, it can be observed that:

* There is no discernable trend between the interest rate and the number of public records, home ownership, and length of employment
* The higher the amount of the credit line used relative to total credit available, the higher the interest rate on a loan; same trend with the debt to income ratio   
![Debt-to-Income vs Median Interest Rate](part1_dti_vs_rate.png)
* When an account has been delinquent for 25+ months during the past 2 years, the interest rate is higher
* The rates are slightly higher for people with hardship and those loans not fully paid.
* Loans at risk are function of the average debt-to-income ratio and number of years of employment.
* People with hardships are more riskier.
* Higher average DTI is very risky, whether or not one has hardship, own or rent a home; and it is surprisingly riskier for people without hardship  
![Debt-to-Income vs "At Risk" & "Fully Paid" Borrowers](part1_AtRisk_vs_FullyPaid_dti.png)

From Part 2:
* However, the majority of the loans are distributed around the coast of Africa and South America 
* Women and first timers are less likely to get the loans; with the discrepancy for women much severe. 
* Agriculture and Trade/Commerce account for more than 50% of the Development-Credit-Authority loans; with no specific country or region dominating the recipients.    
![Development_Credit_Authority_DCA](part2_dca_credit_sector.png)