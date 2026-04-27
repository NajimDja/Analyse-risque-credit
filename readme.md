# Risque de défaut de crédit

Le projet a pour but de réaliser une analyse de données, composée de statistiques descriptives univariées, bivariées et multivariées, ainsi des des calculs de corrélation, de lien entre les variables présentes dans le jeu de données, ainsi que la construction et la comparaison de modèles de prédiction/classification sur le risque de défaut de crédit.  

## Les données

Les données proviennent du site [Kaggle](https://www.kaggle.com/datasets/sergionefedov/credit-risk-dataset-50k-loans-10-sectors?select=portfolio_metrics.csv), du fichier "portfolio_metrics.csv", composé de 50 000 prêts uniques et composé de 24 colonnes.

| Column             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| loan_id            | Unique loan identifier                                                      |
| origination_date   | Loan origination date                                                       |
| maturity_date      | Scheduled maturity date                                                     |
| maturity_months    | Tenor in months (12–120)                                                    |
| sector             | Borrower industry sector (10 sectors)                                       |
| loan_type          | term_loan / revolving / mortgage / bond / lease                             |
| collateral         | secured / unsecured / partially_secured                                     |
| initial_rating     | Credit rating at origination (AAA–CCC)                                      |
| credit_score       | Borrower credit score (300–850)                                             |
| ead                | Exposure at Default — outstanding balance ($)                               |
| coupon_rate        | Loan interest rate (%)                                                      |
| leverage           | Debt / EBITDA ratio                                                         |
| interest_coverage  | EBIT / Interest expense                                                     |
| debt_to_equity     | D/E ratio                                                                   |
| pd_annual          | Probability of Default (annual, Basel IRB approach)                         |
| lgd                | Loss Given Default (0–1; accounts for collateral)                           |
| el                 | Expected Loss = PD × LGD × EAD ($)                                          |
| unexpected_loss    | Unexpected Loss = √(PD×(1−PD)) × LGD × EAD ($)                              |
| rwa                | Risk-Weighted Assets (simplified Basel formula, $)                          |
| defaulted          | Binary: 1 if loan defaulted during its life                                 |
| default_date       | Date of default (null if no default)                                        |
| survival_months    | Months until default or maturity                                            |
| recovery_rate      | 1 − LGD (for defaulted loans)                                               |
| loss_given_default | Actual loss = LGD × EAD (for defaulted loans, $)                            |