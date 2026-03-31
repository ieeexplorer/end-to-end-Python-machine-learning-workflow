# end-to-end-Python-machine-learning-workflow
end-to-end Python machine learning workflow that simulates intelligent support-ticket triage. The project includes synthetic data generation, feature engineering, a scikit-learn preprocessing and modeling pipeline, evaluation metrics, model persistence, and JSON/CSV exports designed for downstream automation tools such as n8n. 

* creates or reads support-ticket data
* engineers features
* predicts priority (low, medium, high)
* predicts escalation risk
* exports ready-to-use outputs for dashboards or n8n workflows


ai_ticket_prioritizer/
│
├── smart_queue_ml.py
├── requirements.txt
├── data/
├── outputs/
└── models/

