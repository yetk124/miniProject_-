from flask import Flask, render_template, request, send_from_directory, jsonify
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score


app = Flask(__name__, template_folder="templates", static_folder="static")

# 🔹 MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["SeoulCrimeDB"]

# 🔹 데이터 로드 (MongoDB → Pandas DataFrame)
merged_data = pd.DataFrame(list(db.merge.find()))


# 🔹 초기+선택 화면 (홈페이지)
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 초기 화면 (홈페이지)
@app.route("/main")
def main():
    return render_template("main.html")

# 🔹 데이터 페이지 (MongoDB 데이터 조회)
@app.route("/main/data")
def data():
    data_type = request.args.get("type", "crime")  # 기본값: 'crime'
    
    if data_type == "crime":
        data = list(db.crime.find({}, {"_id": 0}))
    elif data_type == "real_estate_grouped":
        data = list(db.real_estate_grouped.find({}, {"_id": 0}))
    elif data_type == "cctv":
        data = list(db.cctv.find({}, {"_id": 0}))
    elif data_type == "merged":
        data = list(db.merge.find({}, {"_id": 0}))
    elif data_type == "police":
        data = list(db.police.find({}, {"_id": 0}))  # 🔹 경찰서 데이터 추가
    else:
        data = []
    
    return render_template("data.html",
                           data_type=data_type,
                           crime_data=data if data_type == "crime" else None,
                           real_estate_grouped_data=data if data_type == "real_estate_grouped" else None,
                           cctv_data=data if data_type == "cctv" else None,
                           police_data=data if data_type == "police" else None,  # 🔥 경찰서 데이터 전달
                           merged_data=data if data_type == "merged" else None)


# 🔹 차트 메인 페이지
@app.route("/main/map")
def map():
    return render_template("map.html")


@app.route('/main/map2')
def map2():
    return render_template("map2.html")


# 🔹 차트 메인 페이지
@app.route("/main/chart")
def chart():
    return render_template("chart.html")

# 🔹 개별 차트 상세 페이지
@app.route("/main/chart/detail/<image>")
def chart_detail(image):
    title_mapping = {
        "data1.png": "부동산 가격과 범죄율 관계",
        "data2.png": "자치구별 위험도 점수 비교",
        "data3.png": "자치구별 범죄 유형 및 부동산 가격",
        "data4.png": "CCTV 개수와 부동산 가격 관계",
        "data5.png": "안전도, 부동산 가격, 범죄율 3D 분석",
        "data6.png": "자치구별 안전도 점수 비교",
    }

    explain_mapping = {
        "data1.png": """
                <hr>
                <h4>📊 <strong>부동산 가격과 범죄율 관계 (산점도)</strong></h4>
                <ul>
                    <li><strong>X축:</strong> 평균 부동산 거래 금액 (만원)</li>
                    <li><strong>Y축:</strong> 위험도 (범죄율 / CCTV 비율 + 치안시설 비율)</li>
                    <li><strong>색상:</strong> 자치구별 구분</li>
                </ul>
                <h4>📌 <strong>해석:</strong></h4>
                <ul>
                    <li>강남구, 서초구 등은 <strong>부동산 가격이 높지만 위험도도 중간 이상</strong></li>
                    <li>종로구, 중구 등은 <strong>범죄율이 높은 지역</strong></li>
                    <li>가격과 범죄율 간의 직접적인 상관관계는 크지 않음</li>
                </ul>
                <h4>🧐 <strong>결론:</strong></h4>
                <ul>
                    <li>✅ CCTV 개수와 범죄율을 추가적으로 고려해야 함</li>
                    <li>✅ 특정 지역의 범죄 예방 정책이 필요</li>
                </ul>
            """,
    "data2.png": """
            <h3>📌 이미지 설명</h3>
            <h4>📊 <strong>자치구별 위험도 점수 비교 (막대 그래프)</strong></h4>
            <ul>
                <li><strong>X축:</strong> 자치구</li>
                <li><strong>Y축:</strong> 위험도 (범죄 발생 건수 / CCTV 비율 + 치안시설 비율)</li>
                <li><strong>색상:</strong> 위험도 수준 (진한 색 = 위험도가 높음)</li>
            </ul>
            <h4>📌 <strong>해석:</strong></h4>
            <ul>
                <li><strong>종로구, 중구, 용산구, 서초구</strong> 등이 위험도가 높음</li>
                <li><strong>양천구, 성동구, 성북구, 중랑구</strong> 등은 비교적 안전</li>
                <li>위험도가 높은 곳은 <strong>CCTV 부족 또는 범죄 발생 빈도가 높은 지역</strong>일 가능성 있음</li>
            </ul>
            <h4>🧐 <strong>결론:</strong></h4>
            <ul>
                <li>✅ 특정 자치구에 <strong>CCTV 설치 및 치안 강화 필요</strong></li>
            </ul>
        """,

        "data3.png": """
            <h3>📌 이미지 설명</h3>
            <h4>📊 <strong>자치구별 범죄 유형 및 부동산 가격</strong></h4>
            <ul>
                <li><strong>막대 그래프:</strong> 자치구별 범죄 유형별 비율 (살인, 강도, 성범죄, 절도, 폭력)</li>
                <li><strong>꺾은선 그래프:</strong> 부동산 평균 거래 금액</li>
            </ul>
            <h4>📌 <strong>해석:</strong></h4>
            <ul>
                <li>범죄율이 높은 지역과 부동산 가격이 높은 지역이 반드시 일치하지 않음</li>
                <li><strong>종로구, 중구, 용산구</strong>는 범죄율이 높으면서도 부동산 가격이 높은 지역</li>
                <li><strong>성북구, 도봉구, 금천구</strong> 등은 범죄율과 부동산 가격이 모두 낮음</li>
            </ul>
            <h4>🧐 <strong>결론:</strong></h4>
            <ul>
                <li>✅ 범죄 유형별 분석이 추가로 필요</li>
                <li>✅ 특정 범죄 유형이 집중되는 지역은 <strong>별도의 대책 필요</strong></li>
            </ul>
        """,

        "data4.png": """
            <h3>📌 이미지 설명</h3>
            <h4>📊 <strong>CCTV 개수와 부동산 가격 관계 (버블 차트)</strong></h4>
            <ul>
                <li><strong>X축:</strong> CCTV 총 개수</li>
                <li><strong>Y축:</strong> 평균 부동산 거래 금액 (만원)</li>
                <li><strong>버블 크기:</strong> CCTV 개수</li>
                <li><strong>색상:</strong> 평균 거래 금액 (빨강 = 높음, 파랑 = 낮음)</li>
            </ul>
            <h4>📌 <strong>해석:</strong></h4>
            <ul>
                <li><strong>강남구, 서초구, 송파구</strong>는 부동산 가격이 높지만 CCTV 개수는 많지 않음</li>
                <li><strong>중구, 종로구, 동대문구</strong>는 CCTV 개수는 많지만 부동산 가격은 낮음</li>
                <li><strong>강북구, 금천구, 도봉구</strong>는 CCTV 개수도 적고 부동산 가격도 낮음</li>
            </ul>
            <h4>🧐 <strong>결론:</strong></h4>
            <ul>
                <li>✅ CCTV 개수와 부동산 가격의 관계는 단순하지 않음</li>
                <li>✅ CCTV 개수와 범죄율 간의 추가적인 상관분석 필요</li>
            </ul>
        """,

        "data5.png": """
            <h3>📌 이미지 설명</h3>
            <h4>📊 <strong>안전도, 부동산 가격, 범죄율 3D 분석 (3D 산점도)</strong></h4>
            <ul>
                <li><strong>X축:</strong> 범죄율</li>
                <li><strong>Y축:</strong> 부동산 가격</li>
                <li><strong>Z축:</strong> 안전도 점수</li>
                <li><strong>색상:</strong> 부동산 가격 수준 (노랑 = 높음, 보라 = 낮음)</li>
            </ul>
            <h4>📌 <strong>해석:</strong></h4>
            <ul>
                <li><strong>강남구, 서초구, 송파구</strong>는 부동산 가격이 높고, 위험도도 중간 이상</li>
                <li><strong>성북구, 도봉구</strong> 등은 부동산 가격과 위험도가 모두 낮음</li>
                <li>범죄율이 높지만 안전도가 상대적으로 높은 지역도 존재</li>
            </ul>
            <h4>🧐 <strong>결론:</strong></h4>
            <ul>
                <li>✅ "범죄율이 높지만 안전도가 높은 지역"과 "범죄율이 낮지만 안전도가 낮은 지역"을 구분 가능</li>
                <li>✅ 추가적인 분석을 통해 주민 만족도와 치안 정책 고려 필요</li>
            </ul>
        """,

        "data6.png": """
            <h3>📌 이미지 설명</h3>
            <h4>📊 <strong>자치구별 안전도 점수 비교 (막대 그래프)</strong></h4>
            <ul>
                <li><strong>X축:</strong> 자치구</li>
                <li><strong>Y축:</strong> 안전도 점수</li>
                <li><strong>색상:</strong> 안전 수준 (진한 파랑 = 안전도가 낮음, 연한 파랑 = 높음)</li>
            </ul>
            <h4>📌 <strong>해석:</strong></h4>
            <ul>
                <li><strong>양천구, 성동구, 성북구구</strong>는 가장 높은 안전 점수를 기록</li>
                <li><strong>종로구, 중구, 용산구</strong> 등은 상대적으로 안전 점수가 낮음</li>
                <li>안전 점수가 낮은 지역일수록 추가적인 방범 대책이 필요할 수 있음</li>
            </ul>
            <h4>🧐 <strong>결론:</strong></h4>
            <ul>
                <li>✅ 서울 내에서도 자치구별로 큰 차이가 존재하며, 부동산 가격과 안전도가 반드시 일치하지 않음</li>
                <li>✅ 안전도 점수가 낮은 지역에 대해 추가적인 범죄 예방 대책이 필요</li>
            </ul>
    """
    }


    title = title_mapping.get(image, "차트 상세 보기")
    explain = explain_mapping.get(image, "설명 보기")
    return render_template("chart_detail.html", image=image, title=title, explain=explain)

### 📌 범죄 발생 예측 모델
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

def predict_crime():
    # 🔹 데이터 준비
    X = merged_data[["CCTV_총계", "치안시설_합계", "평균거래금액", "전체인구수", "땅면적"]]
    y = merged_data["범죄_합계"]
    
    # 🔹 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔹 데이터 분리 (학습용 80%, 테스트용 20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 🔹 XGBoost 모델 학습
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # 🔹 예측 및 평가 지표 계산
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # 🔹 전체 데이터 예측
    merged_data["예측 범죄 발생 수"] = model.predict(X_scaled)
    merged_data["절대 오차"] = abs(merged_data["예측 범죄 발생 수"] - merged_data["범죄_합계"])

    # 🔹 정렬 및 상위 10개 선택
    result_df = merged_data[["자치구", "범죄_합계", "예측 범죄 발생 수", "절대 오차"]].sort_values(by="절대 오차", ascending=False).head(15)

    # 🔹 HTML 테이블 반환
    return f"""
    <h4>✅ 평균 절대 오차 (MAE): {mae:.2f} 건</h4>
    {result_df.to_html(index=False, classes="styled-table", justify="center")}
    """



### 📌 안전한 거주지 추천 모델
def predict_safety():
    merged_data["안전도_점수"] = merged_data["CCTV_총계"] / (merged_data["범죄_합계"] + 1)
    merged_data["안전도_라벨"] = (merged_data["안전도_점수"] > merged_data["안전도_점수"].median()).astype(int)

    X = merged_data[["CCTV_총계", "치안시설_합계", "평균거래금액", "범죄_합계"]]
    y = merged_data["안전도_라벨"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    new_data = pd.DataFrame({
        "CCTV_총계": [5000, 8000, 6000, 4500, 4900],
        "치안시설_합계": [10, 20, 18, 22, 15],
        "평균거래금액": [80000, 150000, 70000, 90000, 63624],
        "범죄_합계": [3000, 1000, 300, 600, 2400]
    })
    
    new_predictions = clf.predict(new_data)
    new_data["예측_안전도"] = ["안전" if pred == 1 else "위험" for pred in new_predictions]

     # 🔹 HTML 테이블로 변환
    result_html = new_data.to_html(index=False, classes="styled-table", justify="center")
    return  result_html


# 🔹 머신러닝 페이지 (예측 결과 출력)
@app.route("/main/ml")
def ml():
    crime_results = predict_crime()
    safety_results = predict_safety()
    return render_template("ml.html", crime_prediction_results=crime_results, safety_prediction_results=safety_results)


if __name__ == "__main__":
    app.run(debug=True)
