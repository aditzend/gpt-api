PRE_PROMPT = """Having these prompts and theis coresponding completions into account:
    

      en movistar me dan wifi 300 megas por $3000, todavia no cerre

      


       {
        "intent": "InformarCompetencia_Baja",
        "entities": [
          {"entity": "ORG","text": "movistar","value":"MOVISTAR"},
          {"entity": "SERVICE","text": "wifi 300 megas","value":"INTERNET_300_MB"},
          {"entity": "PRICE","text": "$3000","value":"3000"},
          {"entity": "RECOVERY_SCORING","text": "","value": 7}
        ]
      }
      

      me cambie a personal me dan 500 mb con un descuento de 30%
      



       {
        "intent": "InformarCompetenciaYDescuento_Baja",
        "entities": [
            {"text":"personal","value":"PERSONAL_FLOW", "entity": "ORG"},
            {"text":"500 mb","value":"INTERNET_500_MB", "entity": "SERVICE"},
            {"text":"30%","value":"30", "entity": "DISCOUNT_PERCENTAGE"},
            {"text":"","value": 5, "entity": "RECOVERY_SCORING"}
          ]
      }

      
      Me ofrecen todo lo mismo en Cablevisión x menos de la mitad de lo que abono acá




       {
        "intent": "InformarCompetenciaYDescuento_Baja",
        "entities": [
            {"text":"Cablevisión Flow","value":"PERSONAL_FLOW", "entity": "ORG"},
            {"text":"todo lo mismo","value":"SAME", "entity": "SERVICE"},
            {"text":"x menos de la mitad","value":"50", "entity": "DISCOUNT_PERCENTAGE"},
            {"text":"","value": 2, "entity": "RECOVERY_SCORING"}
          ]
      }

    What would be the same completion for NEW_PROMPT?

    Only respond with a valid JSON object and nothing else.

    NEW_PROMPT = 'prompt_placeholder'
    """
