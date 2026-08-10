import requests

def get_routing_from_colab(user_query):
    # Điểm đến tĩnh không bao giờ thay đổi
    api_endpoint = "https://nonmature-semitailored-an.ngrok-free.dev/route"
    
    payload = {
        "query": user_query
    }
    
    try:
        # Đặt timeout = 5 giây phòng trường hợp mạng chập chờn
        response = requests.post(api_endpoint, json=payload, timeout=5)
        response.raise_for_status() # Báo lỗi nếu Colab sập
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Không thể kết nối tới Colab Router: {e}")
        # Chỗ này bạn có thể viết cơ chế Fallback (ví dụ: tự động tìm kiếm trên cả 2 luồng)
        return None

# ----- TEST THỰC TẾ -----
query = "Q1. Đoạn tin thời sự nói về việc các nhà hàng đang chặn dòng chảy một con suối cho mục đích du lịch. Một đập nước hoặc đập tạm bằng các tấm ván gỗ được giữ bởi các cọc gỗ thẳng đứng và nhiều thanh chống nghiêng. Sau đoạn này là một người đàn ông áo trắng đứng phỏng vấn. Câu hỏi: chức vụ của người đàn ông này là gì?"

results = get_routing_from_colab(query)

print("KẾT QUẢ TỪ COLAB TRẢ VỀ:")
for item in results:
    print(f"[{item.get('type')}] -> {item.get('text')}")