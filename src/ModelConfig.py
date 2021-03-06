training_schema = {'sku': 'int',
                   'national_inv': 'float',
                   'lead_time': 'float',
                   'in_transit_qty': 'float',
                   'forecast_3_month': 'float',
                   'forecast_6_month': 'float',
                   'forecast_9_month': 'float',
                   'sales_1_month': 'float',
                   'sales_3_month': 'float',
                   'sales_6_month': 'float',
                   'sales_9_month': 'float',
                   'min_bank': 'float',
                   'potential_issue': 'str',
                   'pieces_past_due': 'float',
                   'perf_6_month_avg': 'float',
                   'perf_12_month_avg': 'float',
                   'local_bo_qty': 'float',
                   'deck_risk': 'str',
                   'oe_constraint': 'str',
                   'ppap_risk': 'str',
                   'stop_auto_buy': 'str',
                   'rev_stop': 'str'}

training_impute = {'national_inv': 9,
                   'lead_time': 8,
                   'in_transit_qty': 0,
                   'forecast_3_month': 3,
                   'forecast_6_month': 5,
                   'forecast_9_month': 7,
                   'sales_1_month': 1,
                   'sales_3_month': 3,
                   'sales_6_month': 7,
                   'sales_9_month': 10,
                   'min_bank': 0,
                   'potential_issue': 'no',
                   'pieces_past_due': 0,
                   'perf_6_month_avg': 0.73,
                   'perf_12_month_avg': 0.85,
                   'local_bo_qty': 0,
                   'deck_risk': 'no',
                   'oe_constraint': 'no',
                   'ppap_risk': 'no',
                   'stop_auto_buy': 'yes',
                   'rev_stop': 'no'}

cat_cols = ['deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'potential_issue']

model_features = ['national_inv',
                  'lead_time',
                  'in_transit_qty',
                  'forecast_9_month',
                  'sales_9_month',
                  'min_bank',
                  'pieces_past_due',
                  'perf_6_month_avg',
                  'local_bo_qty',
                  'deck_risk',
                  'ppap_risk',
                  'stop_auto_buy']

model_path = '../model/backorder_logistic.pkl'
feature_transform_path = '../model/feature_transform.pkl'
