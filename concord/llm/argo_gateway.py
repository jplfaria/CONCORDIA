import os, httpx
class ArgoGatewayClient:
    def __init__(self, model="gpto3mini", env="prod", stream=False, user=None):
        base=f"https://apps{'' if env=='prod' else f'-{env}'}.inside.anl.gov/argoapi/api/v1/resource"
        self.url=f"{base}/{'streamchat' if stream else 'chat'}"
        self.model, self.stream = model, stream
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()
        self.headers = {"Content-Type":"application/json"}
        self.cli = httpx.Client(timeout=30)
    def chat(self, prompt: str, system="You are a precise bio‑curator.") -> str:
        if self.model.startswith("gpto"):
            payload = {"user":self.user,"model":self.model,"prompt":[prompt],"max_completion_tokens":1024}
        else:
            payload = {"user":self.user,"model":self.model,
                       "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
                       "temperature":0.0}
        r=self.cli.post(self.url,json=payload,headers=self.headers); r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
def llm_label(a,b,client): 
    prompt=(f"Classify the relationship between these two gene annotations.\n"
            f"Return one of: Identical, Synonym, Partial, New.\nA:{a}\nB:{b}")
    return client.chat(prompt).split()[0]
