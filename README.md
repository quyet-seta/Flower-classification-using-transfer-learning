# Flower-classification-using-transfer-learning
## Dataset
- Tâp dữ liệu bao gồm 1360 điểm dữ liệu tương ứng với 17 loài hoa https://drive.google.com/file/d/1MCednjgekCMctCKV0bOIublMwLizuJck/view?usp=sharing
## Classification Model
- Dữ liệu được qua mô hình VGG16 để pre-train cho output là các đặc trưng của ảnh
- ![image](https://user-images.githubusercontent.com/71560376/133905781-53ccca05-2f75-4fbe-8429-ef6778aeed63.png)
- Sau đó qua mô hình phân loại Logistic Regression chiến lược one-vs-rest để phân loại
- ![image](https://user-images.githubusercontent.com/71560376/133905826-00f90408-18c2-480e-b572-734289000a7e.png)
## Dependences
1. python 3.x
2. numpy 1.19.5
3. tensorflow 2.5.0
4. sklearn
## Result
![image](https://user-images.githubusercontent.com/71560376/133905865-ec799445-857d-47d7-bc4d-2cecdfcd3fbf.png)

