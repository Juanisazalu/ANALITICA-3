import joblib  ### para guardar modelos

f_final = joblib.load("rf_final.pkl")
m_lreg = joblib.load("m_lreg.pkl")
list_cat=joblib.load("list_cat.pkl")
list_dummies=joblib.load("list_dummies.pkl")
var_names=joblib.load("var_names.pkl")
scaler=joblib.load("scaler.pkl") 

