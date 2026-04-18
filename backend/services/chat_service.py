import os
import google.generativeai as genai
from typing import Optional, Dict, Any

class ChatService:
    """
    Service for interacting with LLM for explanations
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Gemini model"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables. Chat functionality will be limited.")
            self._model = None
            return

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel('gemini-pro')
            print("✅ Gemini model initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Gemini model: {e}")
            self._model = None
            
    def get_explanation(self, user_query: str, context: Dict[str, Any]) -> str:
        """
        Generate an explanation based on the user query and prediction context
        
        Args:
            user_query: The user's question or prompt
            context: Dictionary containing prediction inputs, outputs, and explanations (SHAP)
            
        Returns:
            str: The generated explanation
        """
        if not self._model:
            return "I apologize, but I cannot currently answer your question because the AI service is not configured (missing API key)."
            
        # Construct prompt
        prompt = self._construct_prompt(user_query, context)
        
        try:
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return "I encountered an error while trying to generate an answer. Please try again later."
            
    def _construct_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Construct the system prompt with context"""
        
        return f"""
You are an expert oncologist and data scientist assistant for the AutoPharma project.
Your goal is to explain drug response predictions to users (who may be researchers or medical professionals, or sometimes normal people).

CONTEXT:
User is asking about a prediction for:
- Cell Line: {context.get('cell_line', 'Unknown')}
- Drug: {context.get('drug_name', 'Unknown')}

PREDICTION RESULTS:
- Predicted IC50: {context.get('predicted_ic50', 'N/A')} µM
- Predicted LN(IC50): {context.get('predicted_ln_ic50', 'N/A')}
(Lower IC50 means the drug is MORE effective)

FEATURE IMPORTANCE (SHAP Values) - Top drivers of this prediction:
{self._format_shap_values(context.get('shap_values', {}))}

USER QUESTION:
"{query}"

INSTRUCTIONS:
1. Answer the user's question directly.
2. Use the provided context to support your answer.
3. If the user asks "Why?", use the SHAP values to explain which features contributed most to the prediction. Positive SHAP values push the LN(IC50) higher (less effective), negative values push it lower (more effective).
4. Keep the explanation clear and concave.
5. If the user asks about the biology, use your general knowledge about the cell line or drug if available, but prioritize the specific model output provided.
"""

    def _format_shap_values(self, shap_values: Optional[Dict[str, float]]) -> str:
        if not shap_values:
            return "No feature importance data available."
            
        # Take top 5 and bottom 5 features
        sorted_items = sorted(shap_values.items(), key=lambda x: x[1])
        
        # Most effective (most negative impact on LN(IC50))
        most_effective = sorted_items[:3]
        
        # Least effective (most positive impact on LN(IC50))
        least_effective = sorted_items[-3:]
        
        text = "Features making the drug MORE effective (lowering IC50):\n"
        for k, v in most_effective:
            text += f"- {k}: {v:.4f}\n"
            
        text += "\nFeatures making the drug LESS effective (increasing IC50):\n"
        for k, v in least_effective:
            text += f"- {k}: {v:.4f}\n"
            
        return text
