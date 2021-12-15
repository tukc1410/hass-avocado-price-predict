# import libraries
# pip install pyqt5
from seaborn import widgets
import streamlit as st
import pandas as pd
from sklearn import metrics
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
from math import sqrt
from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot
from streamlit import caching
import matplotlib.pyplot as plt


# Source Code
data = pd.read_csv("avocado.csv")

#--------------
# GUI
st.title("ĐỒ ÁN KHOA HỌC DỮ LIỆU")
st.write("# Ứng dụng dự báo giá bơ Hass")


# PART 2. Filter Organic Avocado - California
# Make new dataframe from original dataframe: data
df_2=data.loc[data['region']=="California"]
df_2=df_2.loc[df_2['type']=="organic"]
df_2=df_2[['Date','AveragePrice']]
df_2.index=df_2['Date']
df_2=df_2[['AveragePrice']]
df_2.sort_index(inplace=True)
df_2.index=pd.to_datetime(df_2.index)
df_2_new=df_2.reset_index()
df_2_new.columns=['ds','y']
result = seasonal_decompose(df_2, model='multiplicative')
# Train/Test Prophet


# PART 3. Filter Conventional Avocado - California      
df_5=data.loc[data['region']=="California"]
df_5=df_5.loc[df_5['type']=="conventional"]
df_5=df_5[['Date','AveragePrice']]
df_5.index=df_5['Date']
df_5=df_5[['AveragePrice']]
df_5.sort_index(inplace=True)
df_5.index=pd.to_datetime(df_5.index)
df_5_new=df_5.reset_index()
df_5_new.columns=['ds','y']
result5 = seasonal_decompose(df_5, model='multiplicative')
        



# PART 4. Organic avocado in SanDiego
# Make new dataframe from original dataframe: data
df_6=data.loc[data['region']=="SanDiego"]
df_6=df_6.loc[df_6['type']=="organic"]
df_6=df_6[['Date','AveragePrice']]
df_6.index=df_6['Date']
df_6=df_6[['AveragePrice']]
df_6.sort_index(inplace=True)
df_6.index=pd.to_datetime(df_6.index)
df_6_new=df_6.reset_index()
df_6_new.columns=['ds','y']
result6 = seasonal_decompose(df_6, model='multiplicative')

    
# GUI
menu = ["Tổng quan về doanh nghiệp", "Nội dung ứng dụng"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Tổng quan về doanh nghiệp':    
    st.header("Tổng quan về doanh nghiệp")
    st.write("""
    ##### Bơ “Hass”, một công ty có trụ sở tại Mexico, chuyên sản xuất nhiều loại quả bơ được bán ở Mỹ. Họ đã rất thành công trong những năm gần đây và muốn mở rộng. Vì vậy, họ muốn xây dựng mô hình hợp lý để dự đoán giá trung bình của bơ “Hass” ở Mỹ nhằm xem xét việc mở rộng các loại trang trại Bơ đang có cho việc trồng bơ ở các vùng khác.
    """)  
    st.write("""
    ##### => Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ => xem xét việc mở rộng sản xuất, kinh doanh.
    """)
    st.image("image4.jpg")
    st.subheader("Nội dung")
    st.write("""
    #### Phần 1: Tổng quan về bộ dữ liệu
    """)
    st.write("""
    #### Phần 2: Dự báo giá bơ hữu cơ (organic) trung bình ở California
    """)
    st.write("""
    #### Phần 3: Dự báo giá bơ thường (conventional) trung bình ở California
    """)
    st.write("""
    #### Phần 4: Dự báo giá bơ hữu cơ (organic) trung bình ở SanDiego
    """)
    st.write("""
    #### Phần 5: Ứng dụng dự báo giá trung bình bơ Hass
    """)
    st.write("""
     
    """)
    col1, col2, col3 = st.columns([1,6,1])
     
    with col1:
        st.write("")

    with col2:
        st.image(["image.jpg","image5.jpg"])

    with col3:
        st.write("")
    
elif choice=="Nội dung ứng dụng":
    menu2 = ["Tổng quan bộ dữ liệu","Bơ hữu cơ - California", "Bơ thường - California", "Bơ hữu cơ - SanDiego","Ứng dụng người dùng"]
    choice = st.sidebar.selectbox('Nội dung:', menu2)
    if choice=="Tổng quan bộ dữ liệu": 
        st.header("Phần 1. Tổng quan về bộ dữ liệu")  
        st.write(""" #### 1.1. Giá rung bình bơ hữu cơ mắc hơn giá bơ thường trung bình. Giá trung bình (AveragePrice) bị ảnh hưởng bởi loại bơ (type)
        """)
        fig1, ax = plt.subplots(figsize=(20,8))
        sns.boxplot(data=data,x="type",y="AveragePrice")
        plt.show()
        st.pyplot(fig1) 
    
        st.write(""" #### 1.2. Giá trung bình bơ hữu cơ (organic) bị ảnh hưởng bởi khu vực (region)
        """)
        fig2,ax=plt.subplots(figsize=(22,12))
        sns.boxplot(data=data[data['type']=='organic'],
            x="region",y="AveragePrice",ax=ax)
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot(fig2)

        st.write(""" #### 1.3. Giá trung bình bơ thường (conventional) bị ảnh hưởng bởi khu vực (region)
        """)
        fig4,ax=plt.subplots(figsize=(22,12))
        sns.boxplot(data=data[data['type']=='conventional'],
            x="region",y="AveragePrice",ax=ax)
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot(fig4)  

        st.write(""" #### 1.4. Yếu tố tổng sản lượng (Total volume) có tương quan cao với các yếu tố 4046, 4225, 4770, Total Bags, Small Bags, Large Bags.
        """)
        corr=data.corr()
        fig3,ax=plt.subplots(figsize=(10,10))
        sns.heatmap(corr,vmin=-1,vmax=1,annot=True)
        plt.show()
        st.pyplot(fig3) 

    elif choice == 'Bơ hữu cơ - California':            
        menu3 = ["Tổng quan dữ liệu","Model dự báo giá trung bình","Kết quả dự báo"]
        choice = st.sidebar.selectbox('Bơ hữu cơ - California', menu3)
        st.write("""### Phần 2: Dự báo giá bơ hữu cơ trung bình ở California
        
        """)
        if choice=="Tổng quan dữ liệu":           
            st.subheader("""2.1. Tổng quan giá trung bình bơ hữu cơ ở California""")
            st.write("""
            ##### Sơ lược dữ liệu:
            """)
            st.dataframe(df_2_new.head(3))
            st.dataframe(df_2_new.tail(3))   
            st.subheader("Trung bình giá trung bình bơ hữu cơ ở California: " + str(round(df_2_new['y'].mean(),2)) + " USD")

            fig5,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_2)
            plt.title("AveragePrice-Organic",color='red',fontsize=20)
            plt.show();
            st.pyplot(fig5)

            
            fig6,ax=plt.subplots(figsize=(9,3))
            result.seasonal.plot()
            plt.show()
            st.pyplot(fig6)

            fig7,ax=plt.subplots(figsize=(9,3))
            result.trend.plot()
            plt.show()
            st.pyplot(fig7)

            st.write("### Giá trung bình bơ hữu cơ ở California có tính thời vụ và có xu hướng tăng, cao nhất trong tháng 9 và thấp nhất trong tháng 3.")
            st.write("### Kết quả MAE để lựa chọn model dự báo: ")
            st.image('Part2-MAE.png')
            st.write("### Arima có MAE thấp nhất, lựa chọn Arima để dự báo giá trung bình bơ hữu cơ (organic) ở California")
        elif choice == 'Model dự báo giá trung bình':
            st.subheader("2.2. Model sử dụng - Arima")  
            st.write("MAE của Arima: 0.138" )
            st.write("""Kết quả MAE của Arima cho thấy model này có khả năng dự báo tốt giá trung bình bơ hữu cơ ở California, MAE = 0.138 (khoảng 8% giá trung bình là 1.69).
                """)
            st.write("##### Trực quan hóa: Giá trung bình và Giá trung bình dự báo từ 04-2017 đến 03-2018 (51 tuần)")
                # Visulaize the result
            st.image('3-Part2.png',width=600)   

        elif choice == 'Kết quả dự báo':
            st.subheader("2.3. Kết quả dự báo giá trung bình Bơ hữu cơ - California - Arima")
            st.write("#### Biểu đồ dự báo 52 tuần (1 năm) tới")
            st.image('1-Part2.png',width=700)

            st.write("#### Biểu đồ dự báo 260 tuần (5 năm) tới")
            # Next 5 years 
            st.image('2-Part2.png',width=700) 
            st.write(""" ### Kết luận : Giá trung bình bơ hữu cơ ở California có xu hướng tăng cả trong ngắn hạn (1 năm) và trong dài hạn (5 năm). Giả định các yếu tố khác là không đổi, xu hướng tăng giá dự báo việc cầu sẽ vượt cung, vì vậy khuyến nghị xem xét việc mở rộng hoạt động sản xuất kinh doanh bơ hữu cơ ở California.
                """)         

    elif choice == 'Bơ thường - California':
        st.write("""
        ### Phần 3: Dự báo giá bơ thường trung bình ở California
        """)
        menu4 = ["Tổng quan dữ liệu","Model dự báo giá trung bình","Kết quả dự báo"]
        choice = st.sidebar.selectbox('Bơ thường - California', menu4)
        if choice=="Tổng quan dữ liệu":
            st.subheader("""3.1. Tổng quan giá trung bình bơ thường ở California""")
            st.write("""
            ##### Sơ lược dữ liệu:
            """)
            st.dataframe(df_5_new.head(3))
            st.dataframe(df_5_new.tail(3))   
            st.subheader("Trung bình giá bơ thường trung bình ở California: " + str(round(df_5_new['y'].mean(),2)) + " USD")

            fig8,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_5)
            plt.title("AveragePrice-Organic",color='red',fontsize=20)
            plt.show();
            st.pyplot(fig8)

            
            fig9,ax=plt.subplots(figsize=(9,3))
            result5.seasonal.plot()
            plt.show()
            st.pyplot(fig9)

            fig10,ax=plt.subplots(figsize=(9,3))
            result5.trend.plot()
            plt.show()
            st.pyplot(fig10)

            st.write("### Giá trung bình bơ thường ở California có tính thời vụ và có xu hướng tăng, cao nhất trong tháng 9 và thấp nhất trong thảng 3.")
            st.write("### Kết quả MAE để lựa chọn model dự báo: ")
            st.image('Part3-MAE.png')
            st.write("### Holtwinters có MAE thấp nhất, lựa chọn Holtwinters để dự báo giá trung bình bơ thường (conventional) ở California")
        elif choice == 'Model dự báo giá trung bình':
            
            st.subheader("3.2. Model sử dụng - Holtwinters")
            st.write("MAE của Holtwinters: 0.152" )
            st.write("""Kết quả này cho thấy model Holtwinters có khả năng dự báo giá trung bình bơ thường ở California, MAE = 0.152 (khoảng 13% so với giá trung bình là 1.11).
                """)
            st.write("##### Trực quan hóa: Giá trung bình và Giá trung bình dự báo từ 04-2017 đến 03-2018 (51 tuần)")
                # Visulaize the result
            st.image('1-Part3.png',width=600) 
        elif choice == 'Kết quả dự báo':
            st.subheader("3.3. Kết quả dự báo giá trung bình Bơ thường - California - Holtwinters")
            st.write("#### Biểu đồ dự báo 52 tuần (1 năm) tới")
            st.image('2-Part3.png',width=700)

            st.write("#### Biểu đồ dự báo 260 tuần (5 năm) tới")
            # Next 5 years 
            st.image('3-Part3.png',width=700)
            st.write(""" ### Kết luận: Giá trung bình bơ thường ở California không có xu hướng tăng trong ngắn hạn (1 năm) và trong dài hạn (5 năm). Giả định các yếu tố khác là không đổi, việc không tăng giá cho thấy rằng cung đã đáp ứng đủ cầu, khuyến nghị không nên mở rộng hoạt động sản xuất kinh doanh bơ thường ở California.
                """)       

    elif choice == 'Bơ hữu cơ - SanDiego':
                  
        menu6 = ["Tổng quan dữ liệu","Model dự báo giá trung bình","Kết quả dự báo"]
        choice = st.sidebar.selectbox('Bơ hữu cơ - SanDiego', menu6)
        st.write("""### Phần 4: Dự báo giá bơ hữu cơ trung bình ở SanDiego
        
        """)
        if choice=="Tổng quan dữ liệu":           
            st.subheader("4.1. Tổng quan giá trung bình bơ hữu cơ ở SanDiego")
            st.write("""
            ##### Sơ lược dữ liệu:
            """)
            st.dataframe(df_6_new.head(3))
            st.dataframe(df_6_new.tail(3))   
            st.subheader("Trung bình giá bơ hữu cơ trung bình ở SanDiego: " + str(round(df_6_new['y'].mean(),2)) + " USD")

            fig27,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_6)
            plt.title("AveragePrice-Organic",color='red',fontsize=20)
            plt.show();
            st.pyplot(fig27)

            
            fig28,ax=plt.subplots(figsize=(9,3))
            result6.seasonal.plot()
            plt.show()
            st.pyplot(fig28)

            fig29,ax=plt.subplots(figsize=(9,3))
            result6.trend.plot()
            plt.show()
            st.pyplot(fig29)

            
            st.write("### Giá trung bình bơ hữu cơ ở SanDiego có tính thời vụ và có xu hướng tăng, cao nhất trong tháng 9 và thấp nhất trong thảng 3.")
            st.write("### Kết quả MAE để lựa chọn model dự báo: ")
            st.image('Part4-MAE.png')
            st.write("### Arima có MAE thấp nhất, lựa chọn Arima để dự báo giá trung bình bơ hữu cơ (organic) ở SanDiego")
        elif choice == 'Model dự báo giá trung bình':
            st.subheader("4.2. Model sử dụng - Arima")
            st.write("MAE: 0.195" )
            st.write("""Kết quả này cho thấy Arima đủ khả năng dự báo tốt giá trung bình bơ hữu cơ ở SanDiego, MAE = 0.195 (khoảng 11% so với giá trung bình là 1.73).
                """)
            st.write("##### Trực quan hóa: Giá trung bình và Giá trung bình dự báo từ 04-2017 đến 03-2018 (51 tuần)")
                # Visulaize the result
            st.image('1-Part4.png',width=600)

        elif choice == 'Kết quả dự báo':
            st.subheader("2.3. Kết quả dự báo giá trung bình Bơ hữu cơ - SanDiego - Arima")
            st.write("#### Biểu đồ dự báo 52 tuần (1 năm) tới")
            st.image('2-Part4.png',width=700)

            st.write("#### Biểu đồ dự báo 260 tuần (5 năm) tới")
            # Next 5 years 
            st.image('3-Part4.png',width=700) 
            st.write(""" ### Kết luận : Giá trung bình bơ hữu cơ ở SanDiego có xu hướng tăng cả trong ngắn hạn (1 năm) và trong dài hạn (5 năm). Giả định các yếu tố khác là không đổi, xu hướng tăng giá dự báo việc cầu sẽ vượt cung, vì vậy khuyến nghị xem xét việc mở rộng hoạt động sản xuất kinh doanh bơ hữu cơ ở SanDiego.
                """) 
    elif choice=="Ứng dụng người dùng":
        st.header("Ứng dụng dự báo giá")
        st.write("#### Mời bạn chọn khu vực, loại bơ, mô hình muốn dùng cho dự báo giá trung bình bơ Hass")
        st.warning("Vui lòng ấn chọn đủ 3 trường cho mỗi lần muốn dự báo")
        def region():
            return st.selectbox("Regions:",
                    ['','Northeast','Southeast','NorthernNewEngland','SouthCentral','RaleighGreensboro','Detroit','California','Columbus',
                    'NewYork','StLouis','HarrisburgScranton','LosAngeles','GrandRapids','Boise','Seattle','Atlanta','Chicago','Portland',
                    'RichmondNorfolk','LasVegas','Pittsburgh','Houston','SanFrancisco','NewOrleansMobile','TotalUS','HartfordSpringfield',
                    'Denver','Louisville','Boston','Indianapolis','Albany','PhoenixTucson','SanDiego','Plains','Tampa','SouthCarolina',
                    'West','Roanoke','BaltimoreWashington','Charlotte','Midsouth','Jacksonville','GreatLakes','Orlando','DallasFtWorth',
                    'CincinnatiDayton','Spokane','Syracuse','Philadelphia','MiamiFtLauderdale','BuffaloRochester','Sacramento','WestTexNewMexico'])
        region=region() 

        def type():            
            return st.selectbox("Type:",
                                ['','organic','conventional'])
        type=type()


        def model():
            return st.selectbox("Model:",
                        ['','facebook prophet','holtwinters'])
        model=model()
        if (region!="" and type!="" and model=="facebook prophet"):
            df_4=data.loc[data['region']==region]
            df_4=df_4.loc[df_4['type']==type]
            df_4=df_4[['Date','AveragePrice']]
            df_4.index=df_4['Date']
            df_4=df_4[['AveragePrice']]   
            df_4.sort_index(inplace=True)
            df_4.index=pd.to_datetime(df_4.index)
            df_4_new=df_4.reset_index()
            df_4_new.columns=['ds','y']
            result4 = seasonal_decompose(df_4, model='multiplicative')
            # Train/Test Prophet
            train4,test4=np.split(df_4_new,[int(0.7*len(df_4_new))])
            # Build model
            @st.cache(suppress_st_warning=True)
            def load_model4():
                model4=Prophet() 
                model4.fit(train4)  
                weeks4=pd.date_range('2017-04-09','2018-03-25',freq='W').strftime("%Y-%m-%d").tolist() # thời gian của test để so sánh yhat và ytest
                future4=pd.DataFrame(weeks4)
                future4.columns=['ds']
                future4['ds']=pd.to_datetime(future4['ds']) 
                return model4.predict(future4)
            forecast4=load_model4()
            df_4_new.y.mean()
            # 51 weeks in test 
            test4.y.mean()
            y_test4=test4['y'].values
            y_pred4=forecast4['yhat'].values[:51]
            mae_p4=mean_absolute_error(y_test4,y_pred4)
            rmse_p4=sqrt(mean_squared_error(y_test4,y_pred4))
            # Long-term prediction for the next 1-5 years => Consider whether to expand cultivation/production, and trading
            y_test_value4=pd.DataFrame(y_test4,index=pd.to_datetime(test4['ds']),columns=['Actual'])
            y_pred_value4=pd.DataFrame(y_pred4,index=pd.to_datetime(test4['ds']),columns=['Prediction'])
            #predict for next 52 weeks (1 year)
            m5=Prophet()
            m5.fit(df_4_new)
            @st.cache(suppress_st_warning=True)
            def load_model5():
                future_14=m5.make_future_dataframe(periods=52,freq='W')
                return m5.predict(future_14)
            forecast_14=load_model5()
            #predict for next 260 weeks (5 years)
            @st.cache(suppress_st_warning=True)
            def load_model6():
                future_24=m5.make_future_dataframe(periods=260,freq='W')
                return m5.predict(future_24)
            forecast_24=load_model6()
        st.warning("Vui lòng ấn nút bên dưới để xóa lịch sử bộ nhớ trước khi dự báo")
        if st.button("Xóa lịch sử bộ nhớ"):
            st.legacy_caching.clear_cache()

        if st.button("Dự báo"):
            st.legacy_caching.clear_cache()
      
        if (region!="" and type!="" and model=="facebook prophet"):
            st.header("1.Tổng quan giá trung bình bơ "+type +" (organic-hữu cơ/conventional-thường) ở " +region)
            fig13,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_4)
            plt.title("AveragePrice-"+type+"-"+region,color='red',fontsize=20)
            plt.show();
            st.pyplot(fig13)
 
            fig14,ax=plt.subplots(figsize=(9,3))
            result4.seasonal.plot()
            plt.show()
            st.pyplot(fig14)

            fig15,ax=plt.subplots(figsize=(9,3))
            result4.trend.plot()
            plt.show()
            st.pyplot(fig15)
    
            st.subheader("Trung bình giá trung bình bơ " +type+ " (organic-hữu cơ/conventional-thường) ở "+ region+": " + str(round(df_4_new['y'].mean(),2))+ "USD")
            
            st.header("2.Model sử dụng - Facebook Prophet")
            st.subheader("MAE: " + str(round(mae_p4,2)))
            st.write("##### Trực quan hóa: Giá trung bình và Giá trung bình dự báo từ 04-2017 đến 03-2018 (51 tuần)")
            # Visulaize the result
            fig16, ax = plt.subplots()   
            plt.plot(y_test_value4,label='Real')
            plt.plot(y_pred_value4,label='Prediction')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()   
            st.pyplot(fig16)  
            st.header("3.Kết quả dự báo cho " +region+" - "+type+"(organic-hữu cơ/conventional-thường) bơ - Facebook Prophet")
            st.write("##### Biểu đồ dự báo 52 tuần (1 năm) tới")            
            # Next 1 years   
            fig17=m5.plot(forecast_14)
            fig17.show()
            a=add_changepoints_to_plot(fig17.gca(),m5,forecast_14)
            st.pyplot(fig17)

            fig18, ax = plt.subplots()
            plt.plot(df_4_new['y'],label='AveragePrice')
            plt.plot(forecast_14['yhat'],label='Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig18)
    
            st.write("##### Biểu đồ dự báo 260 tuần (5 năm) tới")
            # Next 5 years   
            fig19=m5.plot(forecast_24)
            fig19.show()
            a=add_changepoints_to_plot(fig19.gca(),m5,forecast_24)
            st.pyplot(fig19)

            fig20, ax = plt.subplots()
            plt.plot(df_4_new['y'],label='AveragePrice')
            plt.plot(forecast_24['yhat'],label='Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig20)


        if (region!="" and type!="" and model=="holtwinters"):
            df_3_new=data.loc[data['region']==region]
            df_3_new=df_3_new.loc[df_3_new['type']==type] 
            df_3_new=df_3_new[['Date','AveragePrice']]
            df_3_new.index=df_3_new['Date']
            df_3_new=df_3_new[['AveragePrice']]
            df_3_new.sort_index(inplace=True)
            df_3_new.index=pd.to_datetime(df_3_new.index)
            result2 = seasonal_decompose(df_3_new, model='multiplicative')
            train3,test3=np.split(df_3_new,[int(0.7*len(df_3_new))])
            model3 = ExponentialSmoothing(train3, seasonal='mul', 
                             seasonal_periods=52).fit()

            @st.cache(suppress_st_warning=True)
            def load_model7():    
                return model3.predict(start=test3.index[0], 
                     end=test3.index[-1])
            pred3=load_model7()
            mae3 = mean_absolute_error(test3,pred3)
            rmse3=mean_squared_error(test3,pred3)
            #predict for next 52 weeks (1 year)
            import datetime
            @st.cache(suppress_st_warning=True)
            def load_model8():   
                s = datetime.datetime(2018, 3, 25)
                e = datetime.datetime(2019, 3,  24)
                return model3  .predict(start= s, end=e)
            x=load_model8()
            pred_next_52_week3=x[1:]

            next_52_week3=x.index[1:]
            values_next_52_week3=x.values[1:]
            #predict for next 260 weeks (5 years)
            @st.cache(suppress_st_warning=True)
            def load_model9():  
                s2 = datetime.datetime(2018, 3, 25)
                e2 = datetime.datetime(2023, 3,  19)
                return model3.predict(start= s2, end=e2)
            x2=load_model9()
            pred_next_260_week4=x2[1:]
            next_260_week4=x2.index[1:]
            values_next_260_week4=x2.values[1:] 

        if (region!="" and type!="" and model=="holtwinters"):
            st.header("1.Overview "+type +" avocado in " +region)
            fig21,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_3_new)
            plt.title("AveragePrice-"+type+"-"+region,color='red',fontsize=20)
            plt.show();
            st.pyplot(fig21)

            fig22,ax=plt.subplots(figsize=(9,3))
            result2.seasonal.plot()
            plt.show()
            st.pyplot(fig22)

            fig23,ax=plt.subplots(figsize=(9,3))
            result2.trend.plot()
            plt.show()
            st.pyplot(fig23)
            
            st.subheader("Trung bình giá trung bình bơ " +type+ " (organic-hữu cơ/conventional-thường) ở "+ region+": " + str(round(df_3_new['AveragePrice'].mean(),2)) + " USD")
            st.header("2.Model sử dụng - Holtwinters")
            st.subheader("MAE: " + str(round(mae3,2)))
            st.write("##### Trực quan hóa: Giá trung bình và Giá trung bình dự báo từ 04-2017 đến 03-2018 (51 tuần)")
                # Visulaize the result

            fig24,ax=plt.subplots()
            plt.plot(test3, label='AveragePrice')
            plt.plot(pred3, label='Prediction')
            plt.xticks(rotation='vertical') 
            plt.legend()             
            plt.show()
            st.pyplot(fig24)

            st.header("3.Kết quả dự báo cho " +region+" - "+type+"(organic-hữu cơ/conventional-thường) bơ - Holtwinters")
            st.write("##### Biểu đồ dự báo 52 tuần (1 năm) tới")
   
            fig25,ax=plt.subplots(figsize=(10,6))
            plt.title("AveragePrice from 2015-01 to 2018-03 and next 52 weeks")
            plt.plot(train3.index, train3, label='Train')
            plt.plot(test3.index, test3, label='Test')
            plt.plot(pred3.index, pred3, label='Predict')
            plt.plot(x.index, x.values, label='Next-52-weeks')
            plt.legend(loc='best')            
            st.pyplot(fig25)

            
            st.write("##### Biểu đồ dự báo 260 tuần (5 năm) tới")
            fig26,ax=plt.subplots(figsize=(10,6))
            plt.title("AveragePrice from 2015-01 to 2018-03 and next 260 weeks")
            plt.plot(train3.index, train3, label='Train')
            plt.plot(test3.index, test3, label='Test')
            plt.plot(pred3.index, pred3, label='Predict')
            plt.plot(x2.index, x2.values, label='Next-260-weeks')
            plt.legend(loc='best')
            st.pyplot(fig26)



















            