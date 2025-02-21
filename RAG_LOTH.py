import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ตั้งค่า Gemini API Key
GEMINI_API_KEY = "AIzaSyCFTBU-eaJY9PcQVYrMvDJBDKD0eM3mG7s"  # Gemini API Key
genai.configure(api_key=GEMINI_API_KEY)

# โหลดโมเดลสำหรับแปลงข้อความเป็นเวกเตอร์
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# โหลดโมเดล Gemini
gemini_model = genai.GenerativeModel('gemini-pro')

# อ่านไฟล์ .txt ที่เก็บข้อมูลสถานที่ท่องเที่ยว
file_path = 'location_th.txt'

# อ่านข้อมูลจากไฟล์ .txt
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.readlines()

# จัดระเบียบข้อมูลเป็น Region, Province, และ Tourist Spots
regions = []
provinces = []
tourist_spots = []

current_region = ""
current_province = ""

# ฟังก์ชันที่ตอบคำถามเกี่ยวกับจังหวัดและสถานที่ท่องเที่ยว
def answer_question(query):
    query_lower = query.lower()

    # ถามจังหวัดเพื่อบอกที่เที่ยว
    for province, spots in zip(provinces, tourist_spots):
        if province.lower() in query_lower:
            return f"จังหวัด {province} มีที่เที่ยวดังนี้: {', '.join(spots)}"

    # ถามสถานที่ท่องเที่ยวเพื่อบอกจังหวัด
    for province, spots in zip(provinces, tourist_spots):
        for spot in spots:
            if spot.lower() in query_lower:
                return f"{spot} อยู่ที่จังหวัด {province}"

    # คำถามเกี่ยวกับประเภทสถานที่ท่องเที่ยว
    if "ทะเล" in query_lower:
        return "สถานที่ท่องเที่ยวทะเลที่น่าสนใจ เช่น เกาะสมุย, หาดป่าตอง, เกาะพีพี"
    if "ภูเขา" in query_lower:
        return "สถานที่ท่องเที่ยวภูเขาที่น่าสนใจ เช่น ดอยสุเทพ, เขาค้อ, ดอยอินทนนท์"

    return "ไม่พบข้อมูลที่ตรงกับคำถาม ลองถามใหม่อีกครั้ง!"

# อ่านข้อมูลจากไฟล์และจัดเรียง
for line in data:
    line = line.strip()
    
    # ถ้าเจอชื่อภาค
    if line.endswith("ภาค"):
        current_region = line
    # ถ้าเป็นชื่อจังหวัด
    elif ":" in line:
        current_province, spots = line.split(":")
        current_province = current_province.strip()
        spots = spots.strip().split(",")  # แยกสถานที่ท่องเที่ยว
        regions.append(current_region)
        provinces.append(current_province)
        tourist_spots.append(spots)

# สร้าง FAISS vector store สำหรับค้นหาข้อมูล
class FAISSStore:
    def __init__(self, texts):
        self.texts = texts
        self.embeddings = self.create_embeddings(texts)
        self.index = self.create_index(self.embeddings)
    
    def create_embeddings(self, texts):
        # แปลงข้อความเป็น embeddings
        return np.array([self.text_to_embedding(text) for text in texts])

    def text_to_embedding(self, text):
        # ใช้โมเดลในการแปลงข้อความเป็นเวกเตอร์
        return model.encode(text).tolist()

    def create_index(self, embeddings):
        index = faiss.IndexFlatL2(len(embeddings[0]))  # สร้าง index ด้วย FAISS
        index.add(np.array(embeddings, dtype=np.float32))
        return index

    def similarity_search(self, query, k=3):
        query_embedding = self.text_to_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return [self.texts[i] for i in indices[0]]

# สร้าง FAISS store ด้วยข้อมูลสถานที่ท่องเที่ยวทั้งหมด
texts = [f"{region} - {province}: {', '.join(spots)}" for region, province, spots in zip(regions, provinces, tourist_spots)]
vectorstore = FAISSStore(texts)

def ask_question(question):
    # ค้นหาข้อมูลที่เกี่ยวข้อง
    relevant_docs = vectorstore.similarity_search(question, k=100)
    context = " ".join(relevant_docs)

    # ถ้าไม่มีข้อมูลที่เกี่ยวข้อง
    if not context.strip():
        return "ฉันไม่ทราบคำถามของคุณครับ"

    # สร้างคำถามและคำตอบจาก Gemini API
    prompt = f"""
คำถาม: {question}
ข้อมูลที่เกี่ยวข้อง: {context}

โปรดตอบคำถามตามข้อมูลที่มีให้ โดยพยายามตีความคำถามของผู้ใช้แม้คำถามอาจไม่สมบูรณ์ โดยเน้นตอบตรงประเด็น:

1. หากคำถามถามเกี่ยวกับสถานที่ท่องเที่ยวในจังหวัดใด ให้บอกสถานที่ท่องเที่ยวในจังหวัดนั้น ๆ
2. หากคำถามถามเกี่ยวกับสถานที่ท่องเที่ยวในภาคต่าง ๆ ให้แนะนำสถานที่ท่องเที่ยวในภาคนั้น ๆ
3. หากคำถามถามเกี่ยวกับประเภทสถานที่ เช่น ทะเล, ภูเขา หรือเมืองเก่า ให้ตอบสถานที่ท่องเที่ยวที่สอดคล้องกับคำถามนั้น ๆ
4. หากคำถามไม่ได้ระบุรายละเอียดชัดเจน ให้แนะนำสถานที่ท่องเที่ยวในภาคหรือจังหวัดที่ใกล้เคียงกับคำถาม
5. หากคำถามไม่สามารถตีความได้จากข้อมูล ให้ตอบว่า "ฉันไม่ทราบคำถามของคุณครับ"
6. ตอบคำถามให้กระชับและตรงประเด็น
"""

    # สร้างคำตอบจาก Gemini
    response = gemini_model.generate_content(prompt)

    # ถ้าคำตอบมีคำว่า "ไม่ทราบ" ให้แสดงแค่ "ไม่ทราบ"
    if "ฉันไม่ทราบคำถามของคุณครับ" in response.text:
        return "ฉันไม่ทราบคำถามของคุณครับ"

    # ให้คำตอบที่แม่นยำและกระชับ
    return response.text.strip()

# เริ่มต้นการถามตอบต่อเนื่อง
while True:
    question = input("ถามคำถามเกี่ยวกับที่เที่ยว: ")
    if question.lower() == "exit":
        print("ออกจากระบบการถามตอบ")
        break
    
    answer = ask_question(question)
    print("คำตอบ:", answer)
