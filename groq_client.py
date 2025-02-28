from config import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any, List

class GroqClient:
    def __init__(self):
        self.model = ChatGroq(
            temperature=0, 
            model="llama-3.3-70b-versatile", 
            api_key=GROQ_API_KEY
        )

    def extract_info(self, query: str, history: List[str]) -> Dict[str, Any]:
        # extract structured information from a customer query using Groq LLM
        # info like product categoty, priority, urgency, actions by customer, issue, intent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze the provided customer/user query and extract the following information:
            - Product Category (`query_product`)
            - Priority (Low, Medium, High) (`query_priority`)
            - Urgency (Low, Medium, High) (`query_urgency`)
            - Action Done by Customer (`customer_actions`)
            - Intent (`customer_intent`)
            - Issue (`customer_issue`)
            - Decision (`decision`) (default: auto-respond)
             
            Instructions:
            - Also use the provided history to determine the attributes
            - If the history contains repeated queries then decide to `escalate` else you can auto-respond.
             
            Past Conversation History:
            {history}

            Provide ONLY a valid JSON response with the exact keys mentioned above. 
            Example output:
            {{
                "query_product": "Laptop",
                "query_priority": "High", 
                "query_urgency": "High",
                "customer_actions": "Tried restarting",
                "customer_intent": "Technical support",
                "customer_issue": "Won't turn on",
                "decision": "auto-respond"
            }}
            """),
            ("human", "Query: {query}")
        ])

        chain = prompt | self.model | JsonOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "history": "\n".join(history) if history else "No previous conversation history."
            })
            print('[DEBUG] LLM Chain: ', result)
            return result
        except Exception as e:
            print("[ERROR] LLM Chain: ", str(e))
            # Return a default structured response if parsing fails
            return {
                "query_product": None,
                "query_priority": None, 
                "query_urgency": None,
                "customer_actions": None,
                "customer_intent": None,
                "customer_issue": None,
                "decision": "escalate"
            }

    def generate_final_response(self, query: str, context: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        # Generate a final response based on the query, context, and extracted information
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful AI assistant providing technical support. 
            Use the following context and extracted information to generate a precise, helpful response.
            
            Instructions:
             - If the response is not found in the context, then escalate the query to one of our customer delight representatives.
             - It is important to strictly provided answers from the context.
             - Don't escalate if the user query is not clear or has insufficient information. In this scenario, you have to auto respond and ask for more information.

            Knowledge Base/Context:
            ```
             {context}
            ```
             
            Extracted Query Information:
            - Product: {product}
            - Priority: {priority}
            - Urgency: {urgency}
            - Customer Actions: {customer_actions}
            - Customer Intent: {customer_intent}
            - Customer Issue: {customer_issue}

            Original Query: {query}

            Generate a concise, actionable response with high confidence assessment.
            Provide a JSON with two keys:
            1. `answer`: Your detailed support response
            2. `decision`: 'auto-respond' if high confidence, else 'escalate'
            
            Provide ONLY a valid JSON response with the exact keys mentioned above. 
            Example output:
            {{
                "answer": "Here are specific troubleshooting steps for your laptop issue...",
                "decision": "auto-respond"
            }}
            """),
            ("human", "{query}")
        ])

        print('[DEBUG] Prompt: ', prompt.format(query=query, context=context, product=extracted_info.get("query_product", "Unknown"), priority=extracted_info.get("query_priority", "Unknown"), urgency=extracted_info.get("query_urgency", "Unknown"), customer_actions=extracted_info.get("customer_actions", "None specified"), customer_intent=extracted_info.get("customer_intent", "Support"), customer_issue=extracted_info.get("customer_issue", "Unspecified")))

        chain = prompt | self.model | JsonOutputParser()

        try:
            response = chain.invoke({
                "context": context or "No additional context available.",
                "product": extracted_info.get("query_product", "Unknown"),
                "priority": extracted_info.get("query_priority", "Unknown"),
                "urgency": extracted_info.get("query_urgency", "Unknown"),
                "customer_actions": extracted_info.get("customer_actions", "None specified"),
                "customer_intent": extracted_info.get("customer_intent", "Support"),
                "customer_issue": extracted_info.get("customer_issue", "Unspecified"),
                "query": query
            })
            print('[DEBUG] LLM Final Response: ', response)
            return response
        except Exception as e:
            print("[ERROR] Response Generation: ", str(e))
            return {
                "answer": "This is a bummer :(. I'm unable to find a solution at the moment. Please wait while I connect you to one of our customer delight representatives...",
                "decision": "escalate"
            }
