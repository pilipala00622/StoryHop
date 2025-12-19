
import io



src = "data/1_宗亲家的小娘子.txt"
dst = "data/1_宗亲家的小娘子_8000.txt"

with open(src, "r", encoding="utf-8") as f:
    text = f.read(8000)

with open(dst, "w", encoding="utf-8") as f:
    f.write(text)

print("subset chars:", len(text))



