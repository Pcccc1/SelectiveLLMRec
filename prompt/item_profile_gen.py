
SYSTEM_PROMPT = """\
Please help me generate item profile and reasoning for the item.
I will provide you with the title and attributes of the item.

Requirements:
1. Output with the following structure:
{
  "character": "...",
  "description": "...",
  "reasoning": "..."
}
2. The description and reasoning must be no longer than 150 words.
3. The reasoning should explain what kinds of users may like this item.
"""
import json
import requests
import os


LLAMA_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

def llama_chat(system_prompt, user_prompt, temperature=0.7, max_tokens=512):
    payload = {
        "model": "local-llama",   # 名字随便，llama.cpp 不校验
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["</s>"]
    }
    resp = requests.post(
        LLAMA_ENDPOINT,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300
    )
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    return content


class ItemProfileCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Path
    # -------------------------------------------------------------
    def _cache_path(self, item_id: int) -> str:
        return os.path.join(self.cache_dir, f"item_{item_id}.json")

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------
    def load(self, item_id: int):
        path = self._cache_path(item_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # 防止半写入文件导致整个流程炸
            return None

    # -------------------------------------------------------------
    # Save (atomic)
    # -------------------------------------------------------------
    def save(self, item_id: int, data: dict):
        path = self._cache_path(item_id)
        tmp_path = path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 原子替换，防止写一半中断
        os.replace(tmp_path, path)

    # -------------------------------------------------------------
    # Exists
    # -------------------------------------------------------------
    def exists(self, item_id: int) -> bool:
        return os.path.exists(self._cache_path(item_id))



class ItemProfileGenerator:
    def __init__(self, item_profiles, selected_items):
        self.item_profiles = item_profiles
        self.selected_items = selected_items
        self.item_cache = ItemProfileCache(cache_dir="./cache/item_profiles")
    
    def build_item_prompt(self, item_id, name, categories):
        return f"""
            Input:
            {{
            "item_id": "{item_id}",
            "name": "{name}",
            "categories": "{categories}"
            }}
            """



    def generate_item_profiles_llm(
        self,
        save_path=None
    ):
        results = {}

        for item_id in self.selected_items.tolist():

            cached = self.item_cache.load(item_id)
            if cached is not None:
                results[item_id] = cached
                continue
            raw_profile = self.item_profiles[item_id]

            prompt = self.build_item_prompt(
                item_id=item_id,
                name=raw_profile["name"],
                categories=raw_profile.get("categories", "")
            )

            response = llama_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt
            )

            try:
                profile = json.loads(response)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for item {item_id}: {response}")
                continue 
            
            self.item_cache.save(item_id, profile)

            results[item_id] = profile  

        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)



