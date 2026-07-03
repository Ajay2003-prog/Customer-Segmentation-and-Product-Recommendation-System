
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib 
import os



st.set_page_config(page_title= "Retail Analytics Dashboard",
                   page_icon="🛍️",layout="wide")

#Loading Datasets

df=pd.read_csv("cleaned_online_retail.csv")
rfm=pd.read_csv("rfm_with_clusters.csv")
evaluation = pd.read_csv("model_evaluation.csv")
similarity_df = joblib.load("similarity_df.pkl")

Base_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(Base_DIR,"scaler.pkl"))
model = joblib.load(os.path.join(Base_DIR,"kmeans.pkl"))

#creating revenue column

df["revenue"] = df["quantity"]*df["unit_price"]

#cluster columns

cluster_names= {
    0: "Loyal Customers",
    1: "At Risk",
    2: "Champions",
    3: "Regular Customers"
}

rfm["segment"] = rfm["cluster"].map(cluster_names)
# Sidebar Navigation
st.sidebar.title("➤ Navigation")

page = st.sidebar.radio("Go to",[
    "🏡 Dashboard","📈 RFM Analysis", "🛃 Customer Segmentation",
    "📦 Product Recommendation", "🧾 Insights"
])

if page == "🏡 Dashboard":

    st.markdown("""
                <style>
                .main{
                background-color:#f5f7fb;}
                .title{
                text-align:center;
                font-size:38px;
                font-weight:bold;
                color:#1f3b73;}
                .subtitle{
                text-align:center;
                font-size:18px;
                margin-bottom:30px;
                color:gray;}
                .kpi{
                background:white;
                padding:20px;
                border-radius:15px;
                box-shadow:0px 4px 12px rgba(0,0,0,0.18);
                text-align:center;
                margin-bottom:20px;}
                .kpi-title{
                font-size:16px;
                color:#1565CO;
                font-weight:bold;}
                .kpi-value{
                color:#333333;
                font-size:18px;
                font-weight:bold;}
                </style>

                
                """, unsafe_allow_html= True)
    
    st.markdown('<p class = "title"> 🛍️ Customer Segmentation & Product Recommendation Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Retail Analytics using Machine Learning | Google Colab + Streamlit</p>', unsafe_allow_html=True)
    st.write("") 
     
    

    #KPIS

    customers = rfm.shape[0]
    products = df["stock_code"].nunique()
    invoices = df["invoice_no"].nunique()
    revenue = df["revenue"].sum()
    countries = df["country"].nunique()
    score = evaluation["Value"].iloc[0]

    

    

    c1,c2,c3,c4,c5,c6 = st.columns(6)

    with c1:
        st.markdown(f""" <div class = "kpi">
                    <div class = "kpi-title">🛃Customers</div>
                    <div class = "kpi-value">{customers}</div></div>""",
                    unsafe_allow_html= True)
    with c2 :
        st.markdown(f""" <div class = "kpi">
                    <div class = "kpi-title">📦Products</div>
                    <div class = "kpi-value">{products}</div></div>""",
                    unsafe_allow_html= True)    
        
    with c3 :
        st.markdown(f""" <div class = "kpi">
                    <div class = "kpi-title">📄Invoices</div>
                    <div class = "kpi-value">{invoices}</div></div>""",
                    unsafe_allow_html= True)   

    with c4 :
        st.markdown(f""" <div class = "kpi">
                    <div class = "kpi-title">💰Revenue</div>
                    <div class = "kpi-value">{revenue/1000000:.1f}M</div></div>""",
                    unsafe_allow_html= True) 

    with c5 :
        st.markdown(f""" <div class = "kpi">
                   <div class = "kpi-title">🌏Countries</div>
                    <div class = "kpi-value">{countries}</div></div>""",
                    unsafe_allow_html= True)   
        

    with c6 :
        st.markdown(f""" <div class = "kpi">
                   <div class = "kpi-title">💯Silhouette</div>
                    <div class = "kpi-value">{score:.2f}</div></div>""",
                    unsafe_allow_html= True)       
    

    col1 , col2 = st.columns(2)

    with col1 :
        country = (df.groupby("country",as_index=False)["revenue"].sum()
                   .sort_values(by="revenue",ascending=False).head(10))
        fig = px.bar(country,x="country",y="revenue",color="revenue",
                     color_continuous_scale="Blues",title="Top 10 Countries by Revenue")
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        monthly = (df.groupby(df["invoice_date"].dt.to_period("M"))["revenue"].sum()
                   .reset_index())
        monthly["invoice_date"] = monthly["invoice_date"].astype(str)

        fig = px.line(monthly,x="invoice_date",y="revenue",markers=True,
                      title="Monthly Revenue Trend")
        fig.update_traces(line_color="#1565c0")
        fig.update_layout(xaxis_title="Month",yaxis_title="Revenue",height=450)

        st.plotly_chart(fig,use_container_width=True)    
    

    col3 , col4 = st.columns(2)

    with col3:
        products = (df.groupby("description",as_index=False)["revenue"].sum()
                    .sort_values("revenue").head(10))
        fig = px.bar(products,x="revenue",y="description",orientation="h",
                     color="revenue",color_continuous_scale="Viridis",
                     title="Top 10 Products by Revenue")
        fig.update_layout (yaxis={'categoryorder':'total ascending'},height=500)
        st.plotly_chart(fig,use_container_width=True)
    

    with col4:
        cluster = (rfm["cluster"].value_counts().reset_index())
        cluster.columns = ["cluster","customer"]
        fig = px.pie(cluster,names="cluster",values="customer",
                     hole=0.55, color_discrete_sequence= px.colors.qualitative.Set2,
                     title="Customer Segments")
        fig.update_traces(textposition="inside",textinfo="percent+label")
        st.plotly_chart(fig,use_container_width=True)



elif page == "📈 RFM Analysis":
    st.title("📈 RFM Analysis")
    st.markdown("Analyze Customer groups by using RFM analysis and K-Means Clustering.")

    #Charts : Chart-1:Customer Distribution by Cluster(Donut)
    cluster=rfm["segment"].value_counts().reset_index()
    cluster.columns=["segment","customers"]
    fig = px.pie(rfm,names="segment",
                     hole=0.55, color_discrete_sequence= px.colors.qualitative.Set2,
                     title="Customer Distribution by Cluster")
    
    st.plotly_chart(fig,use_container_width=True)


    #Chart-2 : Recency  vs Frequency 

    fig = px.scatter(rfm,x="recency",y="frequency",color="cluster",size="monetary",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     title="Recency vs Frequency")
    st.plotly_chart(fig,use_container_width=True)


    #chart-3 : Monetary by Cluster

    cluster_summary = rfm.groupby("cluster")[["recency","frequency","monetary"]].mean().reset_index()
    cluster_summary["segment"] = cluster_summary["cluster"].map({
        0:"Loyal Customers",
        1:"At Risk",
        2:"champions",
        3:"Regular Customers"
    })

    fig = px.bar(cluster_summary,x="segment",y="monetary",color="segment",
                 title="Average Monetary Value by Customer Segment")
    st.plotly_chart(fig,use_container_width=True)


    #Customer Segment Table
    st.subheader("Customer Segment Summary")
    st.dataframe(rfm.groupby("segment")[["recency","frequency","monetary"]].mean().reset_index()
                 .round(2),hide_index=True)



    
elif page == "🛃 Customer Segmentation":
    st.title("🛃 Customer Segmentation")
    st.markdown("Enter the customer's RFM values to predict their segment.")

    col1,col2,col3 = st.columns(3)
    with col1:
        recency = st.number_input("📅 Recency",min_value=0,value=30)
    with col2:
        frequency = st.number_input("🧺 Frequency",min_value=1,value=5)
    with col3:
        monetary = st.number_input("💰 Monetary",min_value=0.0,value=500.0)

    if st.button("🔍 Predict Segment"):
        new_customer = scaler.transform([[recency,frequency,monetary]])
        prediction = model.predict(new_customer)[0]
        st.success(f"Predicted Cluster : {prediction}")

        if prediction ==0:
            st.info("🌟 Loyal Customers")
        elif prediction == 1 :
            st.info("🔴 At Risk Customers")   

        elif prediction == 2:
            st.info("🟡 Champions") 

        else:
            st.success("🟢 Regular Customers") 




elif page == "📦 Product Recommendation":
    st.title("📦 Product Recommendation")
    st.markdown("Enter the product description to get similar product recommendations.")
    product_list = similarity_df.index.tolist()
    selected_product = st.selectbox("🛍 Select a Product",product_list)
    if st.button("🔍 Recommend Products"):
        similar_products = (similarity_df[selected_product].sort_values(ascending=False).iloc[1:6].
                            reset_index(name="Similarity Score"))
        similar_products.rename(columns={"description":"Product Desctription"},inplace=True)
        similar_products["Similarity Score"] = similar_products["Similarity Score"].round(3)
        st.dataframe(similar_products,hide_index=True,use_container_width=True)


elif page == "🧾 Insights":
    st.title("🧾 Overall Business Insights")

    st.subheader("📊 Key Insights")

    st.success("""
    ✅ Sales analysis shows that a small group of customers contributes significantly to total revenue.

    ✅ RFM analysis identifies Champion, Loyal, Regular, and At-Risk customers based on purchasing behavior.

    ✅ Customer segmentation using K-Means successfully groups customers with similar spending patterns, enabling targeted marketing strategies.

    ✅ Champion customers have the highest purchase frequency and monetary value, while At-Risk customers require retention campaigns.

    ✅ The product recommendation system uses Cosine Similarity to suggest products with similar purchase patterns, improving personalized recommendations.

    ✅ Personalized product recommendations support cross-selling and upselling opportunities, increasing customer engagement.

    ✅ Combining RFM analysis, customer segmentation, and recommendation systems provides a comprehensive understanding of customer behavior and supports data-driven business decisions.
    """)

    st.subheader("💼 Business Recommendations")

    st.info("""
    • Reward Champion customers with loyalty programs and exclusive offers.

    • Re-engage At-Risk customers using personalized discounts and email campaigns.

    • Convert Regular customers into Loyal customers through targeted promotions.

    • Use the recommendation system to suggest related products during customer purchases.

    • Continuously update the customer segmentation and recommendation models as new transaction data becomes available.
    """)


    
    


