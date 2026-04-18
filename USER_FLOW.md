# AutoPharma User Flow

This document details the user journey through the AutoPharma application, highlighting the interaction steps and decision points.

```mermaid
graph TD
    %% Nodes
    Start((Start))
    Landing[Landing Page\n(View Model Metrics)]
    SelectParams[Select Parameters\n(Cell Line & Drug)]
    ClickPredict[Click 'Predict IC50']

    ViewResults[View Results\n(Predicted IC50 & Graphs)]

    CheckActual{Actual Data\nAvailable?}
    ViewComparison[View vs Actual\nComparison Graphs]
    ViewInfo[View Model Info]

    AskAI{Need Explanation?}
    ChatDrAI[Chat with Dr. AI\n(Ask 'Why?', 'Explain')]
    ReadExplanation[Read AI Explanation]

    NewPred[Make New Prediction]

    %% Edges
    Start --> Landing
    Landing --> SelectParams
    SelectParams --> ClickPredict
    ClickPredict --> ViewResults

    ViewResults --> CheckActual
    CheckActual -- Yes --> ViewComparison
    CheckActual -- No --> ViewInfo

    ViewComparison --> ViewInfo
    ViewInfo --> AskAI

    AskAI -- Yes --> ChatDrAI
    ChatDrAI --> ReadExplanation
    ReadExplanation --> AskAI

    AskAI -- No --> NewPred
    NewPred --> SelectParams

    %% Styling
    style Start fill:#22c55e,stroke:#333,stroke-width:2px,color:white
    style ClickPredict fill:#3b82f6,stroke:#333,stroke-width:2px,color:white
    style ChatDrAI fill:#8b5cf6,stroke:#333,stroke-width:2px,color:white
```

## Step-by-Step User Journey

### 1. 🏠 Landing & Overview

- **User Action**: Opens the application.
- **System Display**: Shows the "Drug Response Predictor" dashboard.
- **User Goal**: Reviews the project overview, dataset stats, and current API connection status (`Connected` or `Error`).

### 2. 🔬 Parameter Selection

- **User Action**: Navigates to the **"Select Parameters"** section.
- **Step A**: Selects a **Cancer Cell Line** (e.g., _22RV1_) from the dropdown.
- **Step B**: Selects a **Drug** (e.g., _Bortezomib_) from the dropdown.
- **System Response**: Updates the UI with the selected combination.

### 3. 🔮 Execution

- **User Action**: Clicks the primary **"Predict IC50"** button.
- **System Process**:
  1.  Fetches biological features for the selected cell line.
  2.  Runs the XGBoost model to predict drug sensitivity.
  3.  Calculates SHAP values for interpretability.

### 4. 📊 Results Review

- **User Action**: Reviews the **Prediction Results** section.
- **Key Outputs**:
  - **Predicted IC50**: The drug concentration required for 50% inhibition (lower is better).
  - **Predicted LN(IC50)**: The raw logarithmic prediction.
- **Conditional**: If actual experimental data is available for this pair:
  - **Comparison**: User sees a side-by-side comparison of Predicted vs. Actual values.
  - **Error Metric**: Displays the Absolute Error.

### 5. 🤖 AI Consultation (Dr. AI)

- **trigger**: User wants to understand _why_ the model made this prediction or needs biological context.
- **User Action**: Types a question into the **"Dr. AI Assistant"** chat box (e.g., _"Why is this drug effective for this cell line?"_).
- **System Process**:
  - Sends context (Prediction + SHAP feature importance) to the LLM.
- **User Action**: Reads the AI's explanation, which might highlight specific genes or drug mechanisms.

### 6. 🔄 Iteration

- **User Action**: Decides to test another combination.
- **Flow**: Returns to **Step 2** to select new parameters. Validates hypothesis or searches for better treatments.
