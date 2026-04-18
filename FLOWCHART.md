# AutoPharma System Flowchart

This document outlines the operational flow of the AutoPharma Drug Response Prediction System.

```mermaid
flowchart TD
    %% Nodes
    User([User])
    Frontend[Frontend UI (Streamlit)]

    subgraph Backend_API [FastAPI Backend]
        API_Predict[POST /predict]
        API_Chat[POST /chat]
        DataSvc[DataService]
        ModelSvc[ModelService]
        ChatSvc[ChatService]
        Model[[XGBoost Model]]
        Explainer[[SHAP Explainer]]
    end

    subgraph External_Services [External Services]
        LLM[Google Gemini AI]
    end

    %% Flow
    User -->|1. Selects Cell Line & Drug| Frontend
    User -->|2. Clicks 'Predict'| Frontend

    Frontend -->|3. Request Prediction| API_Predict

    API_Predict -->|4. Get Features| DataSvc
    API_Predict -->|5. Get Model & Scaler| ModelSvc

    DataSvc -->|Load Data| API_Predict
    ModelSvc -->|Predict IC50| Model
    Model -->|Result| API_Predict

    API_Predict -->|6. Calculate Importance| Explainer
    Explainer -->|SHAP Values| API_Predict

    API_Predict -->|7. Return JSON (IC50 + SHAP)| Frontend

    Frontend -->|8. Display Results & Graphs| User

    %% Chat Flow
    User -->|9. Asks Question (Dr. AI)| Frontend
    Frontend -->|10. Send Query + Context| API_Chat

    API_Chat -->|11. Process Request| ChatSvc
    ChatSvc -->|12. Send Prompt + Context| LLM
    LLM -->|13. Return Explanation| ChatSvc
    ChatSvc -->|14. Return Response| API_Chat
    API_Chat -->|15. Display Answer| Frontend

    %% Styling
    style User fill:#f9f,stroke:#333,stroke-width:2px
    style Frontend fill:#e1f5fe,stroke:#0288d1
    style Backend_API fill:#e8f5e9,stroke:#388e3c
    style External_Services fill:#fff3e0,stroke:#f57c00
    style Model fill:#ffccbc,stroke:#d84315
    style Explainer fill:#ffccbc,stroke:#d84315
```

## Detailed Process Description

1.  **User Input**: The user interacts with the Streamlit frontend to select a specific Cancer Cell Line and a Drug compound.
2.  **Prediction Request**: When the user clicks "Predict", the frontend sends a request to the backend.
3.  **Data Retrieval**: The `DataService` looks up the biological features (gene expression, etc.) associated with the selected cell line and drug.
4.  **Inference**:
    - The `ModelService` scales the data.
    - The **XGBoost** model predicts the IC50 value (drug efficacy).
5.  **Explainability**: The **SHAP** explainer calculates which specific features contributed most to this prediction (e.g., "High expression of Gene X lowered the IC50").
6.  **Visualization**: The frontend displays the predicted values and visual comparisons if actual data exists.
7.  **AI Assistance**:
    - The user can ask questions about the result (e.g., "Why is this effective?").
    - The `ChatService` sends the User Query + Prediction Context + SHAP Values to **Google Gemini**.
    - Gemini returns a natural language explanation which is displayed to the user.
