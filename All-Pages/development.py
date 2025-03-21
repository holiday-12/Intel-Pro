import streamlit as st

st.title(":material/info: The Process of Developing a Salary Prediction System for Programmers")
    
st.header("1. Data Preparation")
st.write("""
    ขั้นตอนแรกคือการเตรียมข้อมูล (Data Preparation) โดยทำการโหลดข้อมูลจากไฟล์ survey_results_public.csv ซึ่งเป็นข้อมูลสำรวจของนักพัฒนาซอฟต์แวร์จากหลายประเทศทั่วโลก โดยมีข้อมูลเช่น ประเทศ (Country), ระดับการศึกษา (EdLevel), จำนวนปีที่มีประสบการณ์การเขียนโปรแกรม (YearsCodePro), สถานะการจ้างงาน (Employment) และ เงินเดือนที่ได้รับต่อปี (ConvertedCompYearly)
    
    - การทำความสะอาดข้อมูล:
        - ลบข้อมูลที่มีค่าว่างหรือข้อมูลที่ไม่สมบูรณ์ออก
        - กรองข้อมูลเฉพาะของผู้ที่ "Employed, full-time"
        - กำหนดค่าเงินเดือนให้มีค่าระหว่าง 10,000 ถึง 250,000
        - แทนที่ค่าหมวดหมู่ประเทศที่ไม่พบบ่อยด้วยคำว่า 'other'

    - การจัดกลุ่มข้อมูล:
        - เปลี่ยนแปลงข้อมูลในคอลัมน์ EdLevel (ระดับการศึกษา) โดยแยกเป็นหมวดหมู่ที่สำคัญ เช่น Bachelor's degree, Master's degree, Professional degree, และ Less than a Bachelore.
    """)

st.header("2. Model Algorithms Theory")
st.write("""
    ในขั้นตอนการพัฒนาระบบทำนายเงินเดือนของนักพัฒนาซอฟต์แวร์ เราใช้ 3 อัลกอริธึมหลักในการทำนายเงินเดือน:

    - **Linear Regression**:
        - Linear Regression คืออัลกอริธึมการทำนายที่ใช้ในการประมาณค่าของตัวแปรตาม (Dependent variable) โดยพิจารณาความสัมพันธ์เชิงเส้นกับตัวแปรอิสระ (Independent variable)
        - มันจะสร้างสมการเส้นตรงที่ใช้ในการทำนายค่าของเงินเดือนจากตัวแปรเช่น ประเทศ, ระดับการศึกษา และประสบการณ์

    - **Decision Tree Regressor**:
        - Decision Tree เป็นอัลกอริธึมการทำนายที่ใช้วิธีการแบ่งกลุ่มข้อมูล (Data Splitting) ไปตามลำดับของคำถามหรือกฎที่เกิดขึ้นในข้อมูล โดยมีจุดมุ่งหมายเพื่อหาค่าความเบี่ยงเบนที่น้อยที่สุดในแต่ละกลุ่ม
        - มันจะสร้าง "ต้นไม้" ที่มีการตัดสินใจในแต่ละโหนดเพื่อทำนายค่าของเงินเดือน

    - **Random Forest Regressor**:
        - Random Forest เป็นการรวมกันของ Decision Trees หลาย ๆ ตัว โดยจะมีการสุ่มเลือกชุดข้อมูลและฟีเจอร์ในแต่ละการฝึก (Training) เพื่อสร้างต้นไม้หลายตัว ซึ่งจะนำผลลัพธ์จากหลาย ๆ ต้นไม้มารวมกันเพื่อทำนายผลที่แม่นยำขึ้น
        - อัลกอริธึมนี้มีความแม่นยำสูงในการทำนายเพราะสามารถจัดการกับข้อมูลที่ซับซ้อนได้ดี
    """)

st.header("3. Model Development Process")
st.write("""
    - **การเตรียมข้อมูลและการฝึกสอนโมเดล (Data Preprocessing and Training)**:
        - ข้อมูลที่เตรียมไว้ถูกแยกออกเป็น X (features) และ y (target) ซึ่ง X เป็นตัวแปรที่ใช้ในการทำนาย เช่น ประเทศ, ระดับการศึกษา, จำนวนปีที่มีประสบการณ์ และ y คือเงินเดือน (Salary)
        - Label Encoding ถูกใช้ในการแปลงค่าของข้อมูลที่เป็นข้อความ (เช่น ประเทศ และระดับการศึกษา) ให้เป็นตัวเลขเพื่อให้สามารถนำไปใช้ในโมเดล Machine Learning ได้
        - จากนั้นเราฝึกโมเดล Linear Regression, Decision Tree Regressor และ Random Forest Regressor ด้วยข้อมูลที่ได้จัดเตรียมไว้

    - **การทดสอบและประเมินผล (Model Evaluation)**:
        - การทดสอบโมเดลทำได้โดยการใช้ Mean Squared Error (MSE) หรือ Root Mean Squared Error (RMSE) เพื่อวัดความแม่นยำของโมเดล
        - เราจะคำนวณค่าความผิดพลาดจากการทำนายผลเงินเดือน และแสดงค่าผลลัพธ์ให้ผู้ใช้เห็น เช่น RMSE ที่ได้จาก Linear Regression, Decision Tree และ Random Forest

    - **การบันทึกโมเดล (Model Saving)**:
        - หลังจากที่ได้โมเดลที่เหมาะสมแล้ว โมเดลจะถูกบันทึกลงในไฟล์ save_steps.pkl เพื่อให้สามารถโหลดและใช้งานในภายหลังได้
        - นอกจากนี้ เราจะบันทึก Label Encoder ที่ใช้แปลงค่าของประเทศและระดับการศึกษาด้วย
    """)

st.header("4. Model Deployment using Streamlit")
st.write("""
    การใช้งานโมเดลทำนายเงินเดือนของนักพัฒนาซอฟต์แวร์ถูกพัฒนาในรูปแบบของเว็บแอปพลิเคชันที่ใช้ Streamlit ซึ่งช่วยให้สามารถสร้างอินเทอร์เฟซแบบกราฟิกสำหรับผู้ใช้เพื่อป้อนข้อมูลและรับการทำนายผลจากโมเดลที่พัฒนาได้

    - ผู้ใช้จะกรอกข้อมูลเช่น ประเทศ, ระดับการศึกษา และ จำนวนปีที่มีประสบการณ์ ผ่านอินเทอร์เฟซของ Streamlit
    - เมื่อผู้ใช้คลิกปุ่ม "Calculate Salary" ระบบจะทำการทำนายเงินเดือนโดยใช้โมเดล Linear Regression, Decision Tree และ Random Forest แล้วแสดงผลที่ทำนายออกมาในรูปแบบกราฟแสดงผลลัพธ์จากแต่ละโมเดล
    """)
st.write("""
    ที่มา dataset: https://survey.stackoverflow.co/
    """)

st.write("""
    ที่มา code: https://www.youtube.com/watch?v=xl0N7tHiwlw
    """)