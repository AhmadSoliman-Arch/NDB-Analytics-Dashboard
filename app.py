"""
NDB — Nile Digital Bank
Customer Service Analytics Dashboard
Analytics Types: Descriptive | Diagnostic | Predictive | Prescriptive
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="NDB Analytics Dashboard", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

NAVY='#0A2342'; TEAL='#007B8A'; GOLD='#C9972C'
GREEN='#1A6B3C'; RED='#CC0000'; GRAY='#6B7C93'
PURPLE='#6C3483'; ORANGE='#D35400'

st.markdown(f"""
<style>
  .main{{background-color:#F4F7FA;}}
  h1,h2,h3{{color:{NAVY};}}
  div[data-testid="metric-container"]{{
    background-color:white;border-radius:8px;
    padding:10px 15px;border-left:4px solid {TEAL};
    box-shadow:0 2px 4px rgba(0,0,0,0.08);}}
  .insight-box{{background:white;padding:15px;border-radius:10px;
    border-left:5px solid {TEAL};margin:10px 0;}}
  .pain-box{{background:white;padding:15px;border-radius:10px;
    border-left:5px solid {RED};margin:8px 0;}}
</style>""", unsafe_allow_html=True)

# ── Data Loading ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    db_path = os.path.join(os.path.dirname(__file__), 'NDB_Clean_Dataset', 'NDB_Database.sqlite')
    conn = sqlite3.connect(db_path)
    tbls = {}
    for t in ['customers','accounts','transactions','interactions',
               'complaints','loans','cards','agents','offers','branches','atms']:
        tbls[t] = pd.read_sql(f"SELECT * FROM {t}", conn)
    conn.close()

    def tb(x): return str(x).strip().lower() in ['true','1','yes']

    intr = tbls['interactions']
    intr['Interaction_Hour'] = pd.to_datetime(
        intr['Interaction_Time'], format='%H:%M:%S', errors='coerce').dt.hour
    for col in ['First_Contact_Resolved','Chatbot_Deflectable',
                'Is_Peak_Hour','Repeat_Contact']:
        intr[col] = intr[col].apply(tb)
    for col in ['Cost_Per_Interaction','Wait_Time_Minutes',
                'Handle_Time_Minutes','CSAT_Rating','Repeat_Contact_Count']:
        intr[col] = pd.to_numeric(intr[col], errors='coerce')

    comp = tbls['complaints']
    comp['SLA_Breached'] = comp['SLA_Breached'].apply(tb)
    for col in ['Resolution_Time_Days','Follow_Up_Contacts','Breach_Days']:
        comp[col] = pd.to_numeric(comp[col], errors='coerce')

    agt = tbls['agents']
    for col in ['Avg_Daily_Interactions','Daily_Interaction_Target','FCR_Rate',
                'CSAT_Score','Monthly_Salary_EGP','Overtime_Hours_Monthly',
                'Training_Hours_Completed','Compliance_Score','Escalation_Rate']:
        agt[col] = pd.to_numeric(agt[col], errors='coerce')
    agt['Overload_Ratio'] = agt['Avg_Daily_Interactions'] / agt['Daily_Interaction_Target']
    agt['True_Monthly_Cost'] = agt['Monthly_Salary_EGP'] + (
        agt['Overtime_Hours_Monthly'] * agt['Monthly_Salary_EGP'] / 176 * 1.5)

    br = tbls['branches']
    for col in ['CSAT_Score','FCR_Rate','Avg_Wait_Time_Minutes',
                'Avg_Daily_Footfall','Complaint_Rate']:
        br[col] = pd.to_numeric(br[col], errors='coerce')

    cust = tbls['customers']
    for col in ['Age','CSAT_Score']:
        cust[col] = pd.to_numeric(cust[col], errors='coerce')
    for col in ['Digital_User','Mobile_App_User','WhatsApp_Banking']:
        cust[col] = cust[col].apply(tb)

    txn = tbls['transactions']
    txn['Amount'] = pd.to_numeric(txn['Amount'], errors='coerce')
    txn['Transaction_Hour'] = pd.to_datetime(
        txn['Transaction_Time'], format='%H:%M:%S', errors='coerce').dt.hour
    txn['Is_Peak'] = txn['Transaction_Hour'].apply(
        lambda h: True if h is not None and (11<=h<=13 or 17<=h<=20) else False)

    loans = tbls['loans']
    for col in ['Principal_Amount','Outstanding_Balance',
                'Monthly_Installment','Missed_Payments_Count']:
        loans[col] = pd.to_numeric(loans[col], errors='coerce')

    cards = tbls['cards']
    for col in ['Credit_Limit','Outstanding_Balance']:
        cards[col] = pd.to_numeric(cards[col], errors='coerce')
    for col in ['International_Transactions','Contactless_Enabled','Online_Transactions']:
        cards[col] = cards[col].apply(tb)

    atms = tbls['atms']
    for col in ['Avg_Daily_Transactions','Monthly_Transactions','Uptime_Rate']:
        atms[col] = pd.to_numeric(atms[col], errors='coerce')

    tbls.update({'interactions':intr,'complaints':comp,'agents':agt,
                 'branches':br,'customers':cust,'transactions':txn,
                 'loans':loans,'cards':cards,'atms':atms})
    return tbls

with st.spinner("Loading NDB Analytics..."):
    data = load_data()

intr=data['interactions']; comp=data['complaints']; agt=data['agents']
cust=data['customers']; br=data['branches']; txn=data['transactions']
loans=data['loans']; cards=data['cards']; atms=data['atms']

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:10px;'>
      <h2 style='color:{GOLD};margin:0;'>🏦 NDB</h2>
      <p style='color:white;margin:0;font-size:13px;'>Nile Digital Bank</p>
      <p style='color:{TEAL};margin:0;font-size:11px;'>Analytics Dashboard</p>
    </div><hr style='border-color:{TEAL};'>""", unsafe_allow_html=True)

    page = st.radio("📊 Navigation", [
        "🏠 Executive Overview",
        "😟 PP1 — Agent Overload",
        "⏰ PP2 — Waiting Times",
        "💰 PP3 — Cost Analysis",
        "🎯 PP4 — FCR Analysis",
        "📋 PP5 — Complaints",
        "⚠️ PP6 — Consistency",
        "📊 Descriptive Analytics",
        "🔬 Diagnostic Analytics",
        "🤖 Predictive Analytics",
        "🎮 Prescriptive Analytics",
        "📈 KPI Projections",
        "🗺️ Geographic Analysis",
        "👥 Customer Segmentation",
        "💳 Loan & Card Portfolio",
        "⚙️ ERP & Process Flow",
    ])

    st.markdown(f"""
    <hr style='border-color:{TEAL};'>
    <div style='color:{GRAY};font-size:11px;text-align:center;'>
    </div>""", unsafe_allow_html=True)

# ── Computed globals ────────────────────────────────────────────────
avg_cost   = intr['Cost_Per_Interaction'].mean()
defl_rate  = intr['Chatbot_Deflectable'].mean()
peak_wait  = intr[intr['Is_Peak_Hour']==True]['Wait_Time_Minutes'].mean()
off_wait   = intr[intr['Is_Peak_Hour']==False]['Wait_Time_Minutes'].mean()
fcr_rate   = intr['First_Contact_Resolved'].mean()
sla_comp   = (1-comp['SLA_Breached'].mean())*100
abandon    = (intr['Interaction_Status']=='Abandoned').mean()*100
overload   = agt['Overload_Ratio'].mean()
monthly_v  = 125000
current_m  = monthly_v * avg_cost
after_m    = monthly_v * (defl_rate*4.0 + (1-defl_rate)*avg_cost)
monthly_sv = current_m - after_m
annual_sv  = monthly_sv * 12

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Executive Overview":
    st.markdown(f"<h1 style='color:{NAVY};'>🏦 NDB — Executive Analytics Overview</h1>", unsafe_allow_html=True)
    st.caption("Chatbots for Customer Service Automation | MBA AI in Business | Arab Academy 2026")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Customers","500,000","NDB Scale")
    c2.metric("Monthly Interactions","125,000","Across all channels")
    c3.metric("Avg Cost/Interaction",f"EGP {avg_cost:.2f}","Current baseline")
    c4.metric("FCR Rate",f"{fcr_rate*100:.1f}%","Target: 77.5%")
    c5.metric("SLA Compliance",f"{sla_comp:.1f}%","Target: 87.5%")
    c6.metric("Chatbot Deflectable",f"{defl_rate*100:.1f}%","Opportunity")

    st.divider()
    st.subheader("🔴 6 Critical Pain Points — Current State")
    c1,c2 = st.columns(2)
    pains = [
        ("PP1","High Agent Overload",f"Overload ratio: {overload:.2f}× target | {defl_rate*100:.1f}% deflectable"),
        ("PP2","Long Peak Hour Waits",f"{peak_wait:.1f} min avg peak wait | {abandon:.1f}% abandonment"),
        ("PP3","High Interaction Cost",f"EGP {avg_cost:.2f}/interaction | EGP {current_m/1e6:.2f}M/month"),
        ("PP4","Poor FCR Rate",f"Only {fcr_rate*100:.1f}% resolved on first contact"),
        ("PP5","Complaint Mismanagement",f"SLA compliance: {sla_comp:.1f}% | Avg resolution: {comp['Resolution_Time_Days'].dropna().mean():.1f} days"),
        ("PP6","Service Inconsistency",f"Branch CSAT: {br['CSAT_Score'].min():.1f}–{br['CSAT_Score'].max():.1f} | Agent FCR: {agt['FCR_Rate'].min()*100:.0f}%–{agt['FCR_Rate'].max()*100:.0f}%"),
    ]
    for i,(num,title,metric) in enumerate(pains):
        col = c1 if i%2==0 else c2
        with col:
            st.markdown(f"""<div class='pain-box'>
            <b style='color:{RED};'>{num}</b> <b style='color:{NAVY};'>{title}</b><br>
            <span style='color:{GRAY};font-size:13px;'>{metric}</span></div>""",
            unsafe_allow_html=True)

    st.divider()
    c1,c2,c3 = st.columns(3)
    with c1:
        ch=intr['Interaction_Channel'].value_counts()
        fig=px.pie(values=ch.values,names=ch.index,title="Channel Distribution",hole=0.4,
                   color_discrete_sequence=[NAVY,TEAL,GOLD,GREEN,RED])
        fig.update_layout(height=320,margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sc=intr['Customer_Sentiment'].value_counts()
        cmap={'Positive':GREEN,'Neutral':GRAY,'Negative':GOLD,'Very Negative':RED}
        fig=px.bar(x=sc.index,y=sc.values,title="Customer Sentiment",
                   color=sc.index,color_discrete_map=cmap)
        fig.update_layout(height=320,showlegend=False,
                          margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    with c3:
        hr=intr.groupby('Interaction_Hour').size().reset_index(name='count')
        hr['color']=hr['Interaction_Hour'].apply(
            lambda h:'Peak' if (11<=h<=13 or 17<=h<=20) else 'Normal')
        fig=px.bar(hr,x='Interaction_Hour',y='count',color='color',
                   title="Interactions by Hour",
                   color_discrete_map={'Peak':RED,'Normal':TEAL})
        fig.update_layout(height=320,margin=dict(t=40,b=10,l=10,r=10),
                          legend_title="")
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAIN POINTS 1-6
# ══════════════════════════════════════════════════════════════════
elif page == "😟 PP1 — Agent Overload":
    st.markdown(f"<h2 style='color:{RED};'>😟 Pain Point 1: High Call Center Volume & Agent Overload</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Avg Daily Interactions",f"{agt['Avg_Daily_Interactions'].mean():.0f}",f"Target: {agt['Daily_Interaction_Target'].mean():.0f}")
    c2.metric("Overload Ratio",f"{overload:.2f}×","Above healthy 1.0×",delta_color="inverse")
    c3.metric("Chatbot Deflectable",f"{defl_rate*100:.1f}%","Of all interactions")
    c4.metric("Agent Monthly Cost",f"EGP {agt['True_Monthly_Cost'].sum()/1000:.0f}K","Including overtime")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(agt))),
            y=agt['Avg_Daily_Interactions'].sort_values().values,
            mode='markers',name='Actual',
            marker=dict(color=agt.sort_values('Avg_Daily_Interactions')['Overload_Ratio'],
                       colorscale='RdYlGn_r',size=10,showscale=True)))
        fig.add_hline(y=agt['Daily_Interaction_Target'].mean(),line_color=GREEN,
                      line_dash='dash',annotation_text=f"Target: {agt['Daily_Interaction_Target'].mean():.0f}/day")
        fig.update_layout(title="Agent Daily Interactions vs Target",height=380)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        dv=intr['Chatbot_Deflectable'].mean()
        fig=go.Figure(go.Pie(values=[dv,1-dv],
            labels=['Chatbot Deflectable','Needs Human Agent'],hole=0.55,
            marker_colors=[TEAL,NAVY]))
        fig.add_annotation(text=f"{dv*100:.1f}%",x=0.5,y=0.5,
                           font_size=28,showarrow=False,font_color=TEAL)
        fig.update_layout(title="Chatbot Deflectable Interactions",height=380)
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        dept=agt.groupby('Department').agg(
            Actual=('Avg_Daily_Interactions','mean'),
            Target=('Daily_Interaction_Target','mean')).reset_index()
        fig=go.Figure()
        fig.add_trace(go.Bar(name='Target',x=dept['Department'],y=dept['Target'],marker_color=GREEN))
        fig.add_trace(go.Bar(name='Actual',x=dept['Department'],y=dept['Actual'],marker_color=RED))
        fig.update_layout(title="Interactions by Department: Target vs Actual",
                          barmode='group',height=350,xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        if 'Chatbot_Routing_Decision' in intr.columns:
            rd=intr['Chatbot_Routing_Decision'].value_counts()
            fig=px.bar(x=rd.values,y=rd.index,orientation='h',
                       title="AI Routing Decision Distribution",
                       color=rd.values,color_continuous_scale='RdYlGn')
            fig.update_layout(height=350,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)

elif page == "⏰ PP2 — Waiting Times":
    st.markdown(f"<h2 style='color:{RED};'>⏰ Pain Point 2: Long Waiting Times During Peak Hours</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Overall Avg Wait",f"{intr['Wait_Time_Minutes'].mean():.1f} min")
    c2.metric("Peak Hour Avg Wait",f"{peak_wait:.1f} min",f"+{peak_wait-off_wait:.1f} vs off-peak",delta_color="inverse")
    c3.metric("Abandonment Rate",f"{abandon:.1f}%","During peak hours",delta_color="inverse")
    c4.metric("Post-Chatbot Wait","< 1 min","↓ 88% reduction")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        hourly=intr.groupby('Interaction_Hour').agg(
            Avg_Wait=('Wait_Time_Minutes','mean'),
            Volume=('Interaction_ID','count')).reset_index()
        hourly['Is_Peak']=hourly['Interaction_Hour'].apply(
            lambda h:'Peak' if (11<=h<=13 or 17<=h<=20) else 'Normal')
        fig=px.bar(hourly,x='Interaction_Hour',y='Avg_Wait',color='Is_Peak',
                   color_discrete_map={'Peak':RED,'Normal':TEAL},
                   title="Average Wait Time by Hour of Day")
        fig.add_hline(y=15,line_color=NAVY,line_dash='dash',
                      annotation_text="Pain Threshold: 15 min")
        fig.update_layout(height=380)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        ch_wait=intr.groupby('Interaction_Channel')['Wait_Time_Minutes'].mean().sort_values(ascending=False)
        fig=px.bar(x=ch_wait.index,y=ch_wait.values,
                   color=ch_wait.values,color_continuous_scale='RdYlGn_r',
                   title="Avg Wait Time by Channel")
        fig.update_layout(height=380,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        st=intr['Interaction_Status'].value_counts()
        fig=px.pie(values=st.values,names=st.index,
                   color=st.index,
                   color_discrete_map={'Resolved':GREEN,'Escalated':GOLD,
                                        'Abandoned':RED,'Pending':TEAL},
                   title="Interaction Status Distribution",hole=0.4)
        fig.update_layout(height=340)
        st_obj=st  # rename to avoid conflict
        st_plot=fig
        import streamlit as st2
        st2.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.histogram(intr,x='Wait_Time_Minutes',nbins=30,
                         color='Is_Peak_Hour',
                         color_discrete_map={True:RED,False:TEAL},
                         barmode='overlay',opacity=0.7,
                         title="Wait Time: Peak vs Off-Peak")
        fig.update_layout(height=340)
        import streamlit as stx; stx.plotly_chart(fig,use_container_width=True)

elif page == "💰 PP3 — Cost Analysis":
    st.markdown(f"<h2 style='color:{RED};'>💰 Pain Point 3: High Cost Per Interaction</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Avg Human Cost",f"EGP {avg_cost:.2f}","Per interaction")
    c2.metric("Avg Chatbot Cost","EGP 4.00","Per interaction")
    c3.metric("Monthly Saving",f"EGP {monthly_sv/1000:.0f}K","Post-chatbot")
    c4.metric("Annual Saving",f"EGP {annual_sv/1e6:.1f}M","Projected")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        sc=[('Current\n(100% Human)',current_m),
            (f'With Chatbot\n({defl_rate*100:.0f}% Deflected)',after_m),
            ('Full AI\n(80% Deflected)',monthly_v*(0.8*4.0+0.2*avg_cost))]
        fig=px.bar(x=[s[0] for s in sc],y=[s[1] for s in sc],
                   color=[s[0] for s in sc],
                   color_discrete_sequence=[RED,TEAL,GREEN],
                   title="Monthly Service Cost: Scenarios")
        fig.update_traces(texttemplate='EGP %{y:,.0f}',textposition='inside',
                          textfont_color='white')
        fig.update_layout(height=380,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        ch_cost=intr.groupby('Interaction_Channel')['Cost_Per_Interaction'].agg(['mean','std']).reset_index()
        fig=go.Figure(go.Bar(x=ch_cost['Interaction_Channel'],y=ch_cost['mean'],
            error_y=dict(type='data',array=ch_cost['std'],visible=True),
            marker_color=[NAVY,TEAL,GOLD,GREEN,RED][:len(ch_cost)]))
        fig.update_layout(title="Avg Cost by Channel (±std)",height=380)
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fig=px.histogram(intr,x='Cost_Per_Interaction',nbins=30,
                         title="Cost Distribution",color_discrete_sequence=[NAVY])
        fig.add_vline(x=avg_cost,line_color=GOLD,line_dash='dash',
                      annotation_text=f"Mean: EGP {avg_cost:.2f}")
        fig.update_layout(height=340)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.scatter(agt,x='True_Monthly_Cost',y='Avg_Daily_Interactions',
                       color='Department',size='Overload_Ratio',
                       title="Agent True Cost vs Daily Volume")
        fig.update_layout(height=340)
        st.plotly_chart(fig,use_container_width=True)

elif page == "🎯 PP4 — FCR Analysis":
    st.markdown(f"<h2 style='color:{RED};'>🎯 Pain Point 4: Poor First Contact Resolution (FCR)</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Current FCR Rate",f"{fcr_rate*100:.1f}%","Target: 77.5%",delta_color="inverse")
    c2.metric("Repeat Contact Rate",f"{intr['Repeat_Contact'].mean()*100:.1f}%","Same issue")
    c3.metric("Avg Agent FCR",f"{agt['FCR_Rate'].mean()*100:.1f}%")
    c4.metric("FCR Variance",f"{(agt['FCR_Rate'].max()-agt['FCR_Rate'].min())*100:.1f}pp","Agent range")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        fcr_ch=intr.groupby('Interaction_Channel')['First_Contact_Resolved'].mean()*100
        fig=px.bar(x=fcr_ch.index,y=fcr_ch.values,
                   color=fcr_ch.values,color_continuous_scale='RdYlGn',
                   title="FCR Rate by Channel (%)")
        fig.add_hline(y=fcr_rate*100,line_color=RED,line_dash='dash',
                      annotation_text=f"Current: {fcr_rate*100:.1f}%")
        fig.add_hline(y=77.5,line_color=GREEN,line_dash='dash',
                      annotation_text="Target: 77.5%")
        fig.update_layout(height=380,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fcr_csat=intr.groupby('First_Contact_Resolved')['CSAT_Rating'].mean()
        fig=go.Figure(go.Bar(
            x=['Not Resolved','Resolved'],
            y=[fcr_csat.get(False,0),fcr_csat.get(True,0)],
            marker_color=[RED,GREEN],
            text=[f"{fcr_csat.get(False,0):.2f}",f"{fcr_csat.get(True,0):.2f}"],
            textposition='inside',textfont_color='white',textfont_size=16))
        fig.update_layout(title="CSAT: Resolved vs Not Resolved",
                          yaxis_range=[0,5],height=380)
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        rd=intr['Repeat_Contact_Count'].value_counts().sort_index()
        fig=px.bar(x=rd.index,y=rd.values,
                   color=rd.index,color_discrete_sequence=[GREEN,GOLD,RED,'#8B0000'],
                   title="Repeat Contact Frequency")
        fig.update_layout(height=340,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        if 'Agent_Tier' in agt.columns:
            tier_fcr=agt.groupby('Agent_Tier')['FCR_Rate'].mean()*100
            fig=px.bar(x=tier_fcr.index,y=tier_fcr.values,
                       color=tier_fcr.index,
                       color_discrete_sequence=[GREEN,TEAL,GOLD,RED],
                       title="FCR Rate by Agent Tier")
            fig.update_layout(height=340,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)

elif page == "📋 PP5 — Complaints":
    st.markdown(f"<h2 style='color:{RED};'>📋 Pain Point 5: Ineffective Complaint Management</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Complaints",f"{len(comp):,}")
    c2.metric("SLA Compliance",f"{sla_comp:.1f}%","Target: 87.5%",delta_color="inverse")
    c3.metric("Avg Resolution",f"{comp['Resolution_Time_Days'].dropna().mean():.1f} days","Target: < 3 days")
    c4.metric("Total Follow-Ups",f"{comp['Follow_Up_Contacts'].sum():,.0f}","All avoidable")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        sla_p=comp.groupby('Complaint_Priority')['SLA_Breached'].mean()*100
        for pri in ['Low','Medium','High','Critical']:
            if pri not in sla_p.index: sla_p[pri]=0
        sla_p=sla_p.reindex(['Low','Medium','High','Critical'])
        fig=px.bar(x=sla_p.index,y=sla_p.values,
                   color=sla_p.values,color_continuous_scale='RdYlGn_r',
                   title="SLA Breach Rate by Priority (%)")
        fig.add_hline(y=comp['SLA_Breached'].mean()*100,line_color=NAVY,
                      line_dash='dash',annotation_text="Avg Breach Rate")
        fig.update_layout(height=380,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fu_cat=comp.groupby('Complaint_Category')['Follow_Up_Contacts'].sum().sort_values(ascending=False)
        fig=px.bar(x=fu_cat.values,y=fu_cat.index,orientation='h',
                   title="Follow-Up Contacts by Category (Avoidable)",
                   color=fu_cat.values,color_continuous_scale='Oranges')
        fig.update_layout(height=380,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3=st.columns(3)
    with c1:
        cat=comp['Complaint_Category'].value_counts()
        fig=px.pie(values=cat.values,names=cat.index,title="Complaint Categories",hole=0.4)
        fig.update_layout(height=320)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        res=comp[comp['Resolution_Time_Days'].notna()]
        met=res[~res['SLA_Breached']]['Resolution_Time_Days']
        bre=res[res['SLA_Breached']]['Resolution_Time_Days']
        fig=px.histogram(res,x='Resolution_Time_Days',color='SLA_Breached',
                         color_discrete_map={True:RED,False:GREEN},
                         title="Resolution Time Distribution",nbins=15,opacity=0.8)
        fig.update_layout(height=320)
        st.plotly_chart(fig,use_container_width=True)
    with c3:
        sc=comp['Complaint_Status'].value_counts()
        cmap_s={s:GREEN if s=='Closed' else TEAL if s=='Resolved' else GOLD
                if s=='In Progress' else RED for s in sc.index}
        fig=px.pie(values=sc.values,names=sc.index,color=sc.index,
                   color_discrete_map=cmap_s,title="Complaint Status",hole=0.4)
        fig.update_layout(height=320)
        st.plotly_chart(fig,use_container_width=True)

elif page == "⚠️ PP6 — Consistency":
    st.markdown(f"<h2 style='color:{RED};'>⚠️ Pain Point 6: Inconsistent Service Quality</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Agent FCR Range",f"{agt['FCR_Rate'].min()*100:.0f}% → {agt['FCR_Rate'].max()*100:.0f}%")
    c2.metric("Branch CSAT Range",f"{br['CSAT_Score'].min():.1f} → {br['CSAT_Score'].max():.1f}")
    c3.metric("CSAT Std Dev",f"{br['CSAT_Score'].std():.3f}","Higher=more inconsistent")
    c4.metric("Avoidable Escalations",f"{intr['Escalation_Level'].notna().mean()*100:.1f}%")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        fig=px.scatter(agt,x='Training_Hours_Completed',y='FCR_Rate',
                       color='Compliance_Score',size='Avg_Daily_Interactions',
                       color_continuous_scale='RdYlGn',
                       title="FCR Rate vs Training Hours")
        if 'Agent_Tier' in agt.columns:
            for tier in agt['Agent_Tier'].unique():
                sub=agt[agt['Agent_Tier']==tier]
        fig.update_layout(height=400)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        agt_s=agt.sort_values('FCR_Rate')
        fig=go.Figure(go.Bar(x=agt_s['Agent_ID'],
            y=agt_s['FCR_Rate']*100,
            marker_color=[GREEN if x>=0.5 else GOLD if x>=0.35 else RED
                          for x in agt_s['FCR_Rate']]))
        fig.add_hline(y=agt['FCR_Rate'].mean()*100,line_color=NAVY,
                      line_dash='dash',
                      annotation_text=f"Avg: {agt['FCR_Rate'].mean()*100:.1f}%")
        fig.update_layout(title="FCR Variance Across 45 Agents",height=400,
                          xaxis_tickangle=-90,xaxis_title="")
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        bt=br.groupby('Branch_Type').agg(
            CSAT=('CSAT_Score','mean'),FCR=('FCR_Rate','mean'),
            Wait=('Avg_Wait_Time_Minutes','mean')).reset_index()
        fig=px.bar(bt,x='Branch_Type',y=['CSAT','FCR'],barmode='group',
                   title="Performance by Branch Type",
                   color_discrete_sequence=[TEAL,GOLD])
        fig.update_layout(height=340)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sent_ch=intr.groupby('Interaction_Channel')['Customer_Sentiment'].apply(
            lambda x:(x=='Positive').mean()*100).reset_index()
        sent_ch.columns=['Channel','Positive_Pct']
        fig=px.bar(sent_ch.sort_values('Positive_Pct'),
                   x='Positive_Pct',y='Channel',orientation='h',
                   color='Positive_Pct',color_continuous_scale='RdYlGn',
                   title="Positive Sentiment Rate by Channel (%)")
        fig.update_layout(height=340,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Descriptive Analytics":
    st.markdown(f"<h2 style='color:{NAVY};'>📊 Descriptive Analytics — What Happened?</h2>",unsafe_allow_html=True)
    st.caption("Interaction volumes, channel distributions, transaction patterns, customer segmentation")

    tab1,tab2,tab3 = st.tabs(["🔄 Interactions","💳 Transactions","👥 Customers"])

    with tab1:
        c1,c2,c3=st.columns(3)
        with c1:
            ch=intr['Interaction_Channel'].value_counts()
            fig=px.bar(x=ch.index,y=ch.values,
                       color=ch.values,color_continuous_scale='Blues',
                       title=f"Channel Volume | Total: {len(intr):,}")
            fig.update_layout(height=320,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            uc=intr['Use_Case_Category'].value_counts().head(10)
            fig=px.bar(x=uc.values,y=uc.index,orientation='h',
                       title="Top 10 Use Cases",
                       color=uc.values,color_continuous_scale='Teal')
            fig.update_layout(height=320,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
        with c3:
            hr=intr.groupby('Interaction_Hour').size().reset_index(name='count')
            hr['Peak']=hr['Interaction_Hour'].apply(
                lambda h:'Peak' if (11<=h<=13 or 17<=h<=20) else 'Normal')
            fig=px.bar(hr,x='Interaction_Hour',y='count',color='Peak',
                       color_discrete_map={'Peak':RED,'Normal':TEAL},
                       title="Hourly Distribution")
            fig.update_layout(height=320,legend_title="")
            st.plotly_chart(fig,use_container_width=True)

        c1,c2,c3=st.columns(3)
        c1.metric("Avg Handle Time",f"{intr['Handle_Time_Minutes'].mean():.1f} min")
        c2.metric("Avg Wait Time",f"{intr['Wait_Time_Minutes'].mean():.1f} min")
        c3.metric("Avg CSAT",f"{intr['CSAT_Rating'].mean():.2f}/5.0")

    with tab2:
        c1,c2,c3=st.columns(3)
        with c1:
            tc=txn['Transaction_Channel'].value_counts()
            fig=px.pie(values=tc.values,names=tc.index,title="Transaction Channel",hole=0.4,
                       color_discrete_sequence=[NAVY,TEAL,GOLD,GREEN,RED])
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            tt=txn['Transaction_Type'].value_counts()
            fig=px.bar(x=tt.values,y=tt.index,orientation='h',
                       color=tt.values,color_continuous_scale='Blues',
                       title="Transaction Types")
            fig.update_layout(height=320,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
        with c3:
            hr_t=txn.groupby('Transaction_Hour').size().reset_index(name='count')
            hr_t['Peak']=hr_t['Transaction_Hour'].apply(
                lambda h:'Peak' if h is not None and (11<=h<=13 or 17<=h<=20) else 'Normal')
            fig=px.bar(hr_t,x='Transaction_Hour',y='count',color='Peak',
                       color_discrete_map={'Peak':RED,'Normal':TEAL},
                       title="Transaction Volume by Hour")
            fig.update_layout(height=320,legend_title="")
            st.plotly_chart(fig,use_container_width=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Total Transactions",f"{len(txn):,}")
        c2.metric("Avg Amount",f"EGP {txn['Amount'].mean():,.2f}")
        c3.metric("Peak Transactions",f"{txn['Is_Peak'].mean()*100:.1f}%")

    with tab3:
        c1,c2,c3=st.columns(3)
        with c1:
            seg=cust['Customer_Segment'].value_counts()
            fig=px.pie(values=seg.values,names=seg.index,
                       title="Customer Segments",hole=0.4,
                       color_discrete_sequence=[NAVY,TEAL,GOLD,GREEN])
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.histogram(cust,x='Age',nbins=20,
                             color_discrete_sequence=[TEAL],
                             title=f"Age Distribution | Avg: {cust['Age'].mean():.1f}")
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c3:
            dig={'Digital Users':cust['Digital_User'].sum(),
                 'Mobile App':cust['Mobile_App_User'].sum(),
                 'WhatsApp':cust['WhatsApp_Banking'].sum()}
            fig=px.bar(x=list(dig.keys()),y=list(dig.values()),
                       color=list(dig.values()),color_continuous_scale='Teal',
                       title="Digital Channel Adoption")
            fig.update_layout(height=320,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Diagnostic Analytics":
    st.markdown(f"<h2 style='color:{NAVY};'>🔬 Diagnostic Analytics — Why Did It Happen?</h2>",unsafe_allow_html=True)
    st.caption("Root cause analysis | Correlation matrix | Driver identification")

    tab1,tab2 = st.tabs(["📊 Correlation Matrix","🔍 Root Cause Drilldown"])

    with tab1:
        merged=intr[['Wait_Time_Minutes','Handle_Time_Minutes','Cost_Per_Interaction',
                     'CSAT_Rating','Repeat_Contact','First_Contact_Resolved',
                     'Is_Peak_Hour','Repeat_Contact_Count']].copy()
        merged['Repeat_Contact']=merged['Repeat_Contact'].astype(int)
        merged['First_Contact_Resolved']=merged['First_Contact_Resolved'].astype(int)
        merged['Is_Peak_Hour']=merged['Is_Peak_Hour'].astype(int)
        corr=merged.corr()
        fig=px.imshow(corr,text_auto=True,aspect="auto",
                      color_continuous_scale='RdBu_r',
                      title="Correlation Matrix: Service Metrics")
        fig.update_layout(height=600)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
        <b>📌 Key Diagnostic Insights:</b><br>
        ✅ <b>Wait Time ↔ CSAT</b>: Negative correlation — longer waits directly reduce satisfaction<br>
        ✅ <b>FCR ↔ Repeat Contact</b>: Strong negative — poor FCR drives repeat calls & costs<br>
        ✅ <b>Peak Hour ↔ Cost</b>: Positive — overtime premiums increase cost during peaks<br>
        ✅ <b>Handle Time ↔ Cost</b>: Direct — longer calls = higher agent cost per interaction
        </div>""",unsafe_allow_html=True)

    with tab2:
        c1,c2=st.columns(2)
        with c1:
            peak_ch=intr.groupby(['Interaction_Channel','Is_Peak_Hour']).agg(
                Avg_Wait=('Wait_Time_Minutes','mean')).reset_index()
            peak_ch['Is_Peak_Hour']=peak_ch['Is_Peak_Hour'].map({True:'Peak',False:'Off-Peak'})
            fig=px.bar(peak_ch,x='Interaction_Channel',y='Avg_Wait',
                       color='Is_Peak_Hour',barmode='group',
                       color_discrete_map={'Peak':RED,'Off-Peak':TEAL},
                       title="Wait Time: Peak vs Off-Peak by Channel")
            fig.update_layout(height=360)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            esc_r=intr['Escalation_Reason'].dropna().value_counts().head(8)
            fig=px.bar(x=esc_r.values,y=esc_r.index,orientation='h',
                       color=esc_r.values,color_continuous_scale='Reds',
                       title="Top Escalation Root Causes")
            fig.update_layout(height=360,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            cost_seg=intr.merge(
                cust[['Customer_ID','Customer_Segment']],on='Customer_ID',how='left')
            seg_cost=cost_seg.groupby('Customer_Segment')['Cost_Per_Interaction'].mean().reset_index()
            fig=px.bar(seg_cost,x='Customer_Segment',y='Cost_Per_Interaction',
                       color='Cost_Per_Interaction',color_continuous_scale='RdYlGn_r',
                       title="Avg Cost by Customer Segment")
            fig.update_layout(height=340,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            br_sla=comp.groupby('Complaint_Priority').agg(
                Breach_Rate=('SLA_Breached','mean'),
                Avg_Days=('Resolution_Time_Days','mean')).reset_index()
            fig=px.scatter(br_sla,x='Avg_Days',y='Breach_Rate',
                           size='Breach_Rate',color='Complaint_Priority',
                           title="SLA Breach Rate vs Avg Resolution Days")
            fig.update_layout(height=340)
            st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Analytics":
    st.markdown(f"<h2 style='color:{NAVY};'>🤖 Predictive Analytics — What Will Happen?</h2>",unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["🎯 FCR Prediction","📋 SLA Breach Prediction","⚠️ Churn Risk"])

    with tab1:
        st.subheader("Random Forest — FCR Prediction Model")
        df_ml=intr[['Wait_Time_Minutes','Is_Peak_Hour','Interaction_Channel',
                    'Use_Case_Category','Handle_Time_Minutes','Cost_Per_Interaction',
                    'First_Contact_Resolved']].dropna().copy()
        df_ml['Is_Peak_Hour']=df_ml['Is_Peak_Hour'].astype(int)
        df_ml['First_Contact_Resolved']=df_ml['First_Contact_Resolved'].astype(int)
        le_ch=LabelEncoder(); le_uc=LabelEncoder()
        df_ml['Channel_Enc']=le_ch.fit_transform(df_ml['Interaction_Channel'])
        df_ml['UseCase_Enc']=le_uc.fit_transform(df_ml['Use_Case_Category'])
        feats=['Wait_Time_Minutes','Is_Peak_Hour','Channel_Enc','UseCase_Enc',
               'Handle_Time_Minutes','Cost_Per_Interaction']
        X=df_ml[feats]; y=df_ml['First_Contact_Resolved']
        class_dist=y.value_counts()
        st.caption(f"Dataset: Resolved (1): {class_dist.get(1,0):,} | Not Resolved (0): {class_dist.get(0,0):,}")
        if y.nunique()==1:
            st.warning("Single class detected — showing use case analysis instead")
            uc_c=df_ml['Use_Case_Category'].value_counts().reset_index()
            uc_c.columns=['Use_Case','Count']
            fig=px.bar(uc_c.head(10),x='Count',y='Use_Case',orientation='h',
                       title="High-Volume Use Cases with 0% FCR",
                       color='Count',color_continuous_scale='Reds')
            st.plotly_chart(fig,use_container_width=True)
        else:
            X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
            clf=RandomForestClassifier(n_estimators=100,max_depth=6,
                                       class_weight='balanced',random_state=42)
            clf.fit(X_tr,y_tr)
            y_pred=clf.predict(X_te)
            acc=accuracy_score(y_te,y_pred)
            cv=cross_val_score(clf,X,y,cv=5,scoring='accuracy').mean()
            c1,c2,c3=st.columns(3)
            c1.metric("Model Accuracy",f"{acc:.1%}")
            c2.metric("5-Fold CV Accuracy",f"{cv:.1%}")
            c3.metric("Test Set Size",f"{len(y_te):,}")
            c1,c2=st.columns(2)
            with c1:
                fi=pd.DataFrame({'Feature':['Wait Time','Peak Hour','Channel',
                                            'Use Case','Handle Time','Cost'],
                                 'Importance':clf.feature_importances_}).sort_values('Importance')
                fig=px.bar(fi,x='Importance',y='Feature',orientation='h',
                           color='Importance',color_continuous_scale='Teal',
                           title="Feature Importance — FCR Drivers")
                fig.update_layout(height=380,coloraxis_showscale=False)
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                fcr_pred=pd.DataFrame({
                    'Channel':['Branch','Call Center','WhatsApp','Mobile App','Email'],
                    'Current FCR':[intr[intr['Interaction_Channel']==ch]['First_Contact_Resolved'].mean()*100
                                   for ch in ['Branch','Call Center','WhatsApp','Mobile App','Email']],
                    'Predicted w/ Chatbot':[72,80,78,82,68]})
                x_pos=np.arange(len(fcr_pred))
                fig=go.Figure()
                fig.add_trace(go.Bar(name='Current',x=fcr_pred['Channel'],
                                     y=fcr_pred['Current FCR'],marker_color=RED,opacity=0.85))
                fig.add_trace(go.Bar(name='Predicted w/ Chatbot',x=fcr_pred['Channel'],
                                     y=fcr_pred['Predicted w/ Chatbot'],marker_color=GREEN,opacity=0.85))
                fig.add_hline(y=77.5,line_color=GOLD,line_dash='dash',annotation_text="Target 77.5%")
                fig.update_layout(title="FCR: Current vs Predicted Post-Chatbot",
                                  barmode='group',height=380)
                st.plotly_chart(fig,use_container_width=True)

    with tab2:
        st.subheader("Random Forest — SLA Breach Prediction")
        comp_ml=comp.copy()
        comp_ml['Priority_Enc']=comp_ml['Complaint_Priority'].map(
            {'Low':0,'Medium':1,'High':2,'Critical':3}).fillna(1)
        comp_ml['Channel_Enc']=LabelEncoder().fit_transform(comp_ml['Complaint_Channel'].fillna('Unknown'))
        comp_ml['Cat_Enc']=LabelEncoder().fit_transform(comp_ml['Complaint_Category'].fillna('Unknown'))
        comp_ml['SLA_Breached_Int']=comp_ml['SLA_Breached'].astype(int)
        comp_ml['Escalation_Count']=pd.to_numeric(comp_ml.get('Escalation_Count',0),errors='coerce').fillna(0)
        feats_s=['Priority_Enc','Channel_Enc','Cat_Enc','Escalation_Count','Follow_Up_Contacts']
        df_s=comp_ml[feats_s+['SLA_Breached_Int']].dropna()
        Xs=df_s[feats_s]; ys=df_s['SLA_Breached_Int']
        if ys.nunique()>1:
            clf_s=RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
            clf_s.fit(Xs,ys)
            cv_s=cross_val_score(clf_s,Xs,ys,cv=5,scoring='accuracy').mean()
            c1,c2=st.columns(2)
            c1.metric("SLA Model CV Accuracy",f"{cv_s:.1%}")
            c2.metric("Breach Rate in Data",f"{ys.mean()*100:.1f}%")
            c1,c2,c3=st.columns(3)
            with c1:
                bp=comp.groupby('Complaint_Priority')['SLA_Breached'].mean()*100
                for p in ['Low','Medium','High','Critical']:
                    if p not in bp.index: bp[p]=0
                bp=bp.reindex(['Low','Medium','High','Critical'])
                fig=px.bar(x=bp.index,y=bp.values,color=bp.values,
                           color_continuous_scale='RdYlGn_r',
                           title="Breach Rate by Priority")
                fig.update_layout(height=320,coloraxis_showscale=False)
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                res=comp[comp['Resolution_Time_Days'].notna()]
                fig=px.histogram(res,x='Resolution_Time_Days',color='SLA_Breached',
                                 color_discrete_map={True:RED,False:GREEN},
                                 title="Resolution Time Distribution",nbins=15,opacity=0.8)
                fig.update_layout(height=320)
                st.plotly_chart(fig,use_container_width=True)
            with c3:
                sc_pred=pd.DataFrame({
                    'Scenario':['Current\n(No AI)','With Chatbot\nTracking','With Chatbot +\nRPA'],
                    'SLA Compliance':[sla_comp,80.0,87.5]})
                fig=px.bar(sc_pred,x='Scenario',y='SLA Compliance',
                           color='SLA Compliance',color_continuous_scale='RdYlGn',
                           title="SLA Compliance Prediction")
                fig.update_layout(height=320,coloraxis_showscale=False,yaxis_range=[0,100])
                st.plotly_chart(fig,use_container_width=True)

    with tab3:
        st.subheader("Churn Risk Model — Customer Retention")
        cust_ml=cust.copy()
        cust_ml['Churn_Risk']=((cust_ml['CSAT_Score']<3.0)|
                               (cust_ml['Customer_Status']=='Dormant')).astype(int)
        cust_ml['Segment_Enc']=LabelEncoder().fit_transform(cust_ml['Customer_Segment'].fillna('Retail'))
        cust_ml['Digital_Enc']=cust_ml['Digital_User'].astype(int)
        cust_ml['Mobile_Enc']=cust_ml['Mobile_App_User'].astype(int)
        feats_c=['Age','Segment_Enc','Digital_Enc','Mobile_Enc','CSAT_Score']
        df_c=cust_ml[feats_c+['Churn_Risk']].dropna()
        Xc=df_c[feats_c]; yc=df_c['Churn_Risk']
        clf_c=RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
        clf_c.fit(Xc,yc)
        proba=clf_c.predict_proba(Xc)[:,1]
        cust_ml.loc[df_c.index,'Churn_Probability']=proba
        cv_c=cross_val_score(clf_c,Xc,yc,cv=5,scoring='accuracy').mean()
        c1,c2,c3=st.columns(3)
        c1.metric("Churn Model Accuracy",f"{cv_c:.1%}")
        c2.metric("High Risk Customers",f"{(proba>0.5).sum():,}")
        c3.metric("Avg Churn Probability",f"{proba.mean()*100:.1f}%")
        c1,c2,c3=st.columns(3)
        with c1:
            seg_churn=cust_ml.groupby('Customer_Segment')['Churn_Risk'].mean()*100
            fig=px.bar(x=seg_churn.index,y=seg_churn.values,
                       color=seg_churn.values,color_continuous_scale='RdYlGn_r',
                       title="Churn Risk by Segment (%)")
            fig.update_layout(height=320,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.histogram(x=proba,nbins=20,color_discrete_sequence=[TEAL],
                             title=f"Churn Probability Distribution")
            fig.add_vline(x=0.5,line_color=RED,line_dash='dash',annotation_text="Risk Threshold")
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c3:
            csat_b=pd.cut(cust_ml['CSAT_Score'].dropna(),
                          bins=[0,2,3,4,5],labels=['1-2','2-3','3-4','4-5'])
            churn_b=cust_ml.groupby(csat_b)['Churn_Risk'].mean()*100
            fig=px.bar(x=churn_b.index,y=churn_b.values,
                       color=churn_b.values,color_continuous_scale='RdYlGn_r',
                       title="Churn Risk by CSAT Band")
            fig.update_layout(height=320,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — PRESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "🎮 Prescriptive Analytics":
    st.markdown(f"<h2 style='color:{NAVY};'>🎮 Prescriptive Analytics — What Should We Do?</h2>",unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["🎮 What-If Simulation","🎯 Deployment Optimizer","🌳 AI Routing Engine"])

    with tab1:
        st.subheader("Interactive Chatbot Impact Simulation")
        c1,c2=st.columns([1,2])
        with c1:
            defl_sim=st.slider("Chatbot Deflection Rate (%)",0,100,int(defl_rate*100),5)/100
            chatbot_cost_sim=st.slider("Chatbot Cost/Interaction (EGP)",1,20,4,1)
            vol_sim=st.number_input("Monthly Interaction Volume",50000,200000,125000,5000)
        sim_cost=vol_sim*((defl_sim*chatbot_cost_sim)+((1-defl_sim)*avg_cost))
        sim_saving=(vol_sim*avg_cost)-sim_cost
        ann_saving=sim_saving*12
        with c2:
            c1b,c2b,c3b=st.columns(3)
            c1b.metric("Current Monthly",f"EGP {vol_sim*avg_cost:,.0f}")
            c2b.metric("Simulated Monthly",f"EGP {sim_cost:,.0f}",f"-EGP {sim_saving:,.0f}")
            c3b.metric("Annual Saving",f"EGP {ann_saving:,.0f}")
        fig=go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","total"],
            x=["Current Annual Cost","Chatbot Deployment\n(EGP 3-5M)","Annual\nMaintenance","Annual\nSavings","Net Annual\nBenefit"],
            y=[vol_sim*avg_cost*12,-4000000,-500000,-ann_saving,0],
            decreasing={"marker":{"color":RED}},
            increasing={"marker":{"color":GREEN}},
            totals={"marker":{"color":TEAL}}))
        fig.update_layout(title=f"Annual ROI at {defl_sim*100:.0f}% Deflection Rate",height=420)
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        st.subheader("Use Case Deployment Priority Matrix")
        uc_data=intr.groupby('Use_Case_Category').agg(
            Volume=('Interaction_ID','count'),
            Deflectable=('Chatbot_Deflectable','mean'),
            Avg_Cost=('Cost_Per_Interaction','mean'),
            FCR=('First_Contact_Resolved','mean')).reset_index()
        uc_data['Potential_Saving']=(uc_data['Volume']*uc_data['Deflectable']*
                                     (uc_data['Avg_Cost']-4.0))
        uc_data['Priority_Score']=(uc_data['Deflectable']*0.4+
            (uc_data['Potential_Saving']/uc_data['Potential_Saving'].max())*0.4+
            (1-uc_data['FCR'])*0.2)
        uc_sorted=uc_data.sort_values('Priority_Score',ascending=False)
        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(uc_sorted.sort_values('Priority_Score'),
                       x='Priority_Score',y='Use_Case_Category',orientation='h',
                       color='Priority_Score',color_continuous_scale='RdYlGn',
                       title="Deployment Priority Score (Higher = Deploy First)")
            fig.add_vline(x=0.6,line_color=GREEN,line_dash='dash',annotation_text="High Priority")
            fig.add_vline(x=0.4,line_color=GOLD,line_dash='dash',annotation_text="Medium Priority")
            fig.update_layout(height=450,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.scatter(uc_data,x='Deflectable',y='Potential_Saving',
                           size='Volume',color='Priority_Score',
                           hover_name='Use_Case_Category',
                           color_continuous_scale='RdYlGn',
                           title="Opportunity Matrix (Bubble=Volume)")
            fig.update_layout(height=450)
            st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
        <b>📌 Prescriptive Recommendation:</b><br>
        🟢 <b>Deploy First</b>: {', '.join(uc_sorted.head(3)['Use_Case_Category'].tolist())}<br>
        🟡 <b>Phase 2</b>: {', '.join(uc_sorted.iloc[3:6]['Use_Case_Category'].tolist())}<br>
        ⚪ <b>Phase 3</b>: Remaining use cases after model validation
        </div>""",unsafe_allow_html=True)

    with tab3:
        st.subheader("AI Routing Decision Engine")
        if 'Chatbot_Routing_Decision' in intr.columns:
            rd=intr['Chatbot_Routing_Decision'].value_counts()
            cost_map={'Chatbot Resolves':4.0,'Standard Agent Queue':avg_cost,
                      'Priority Escalation':avg_cost*1.3,'Specialist Route':avg_cost*1.1}
            c1,c2=st.columns(2)
            with c1:
                fig=px.pie(values=rd.values,names=rd.index,
                           title="AI Routing Decision Distribution",hole=0.4,
                           color_discrete_sequence=[GREEN,TEAL,GOLD,RED])
                fig.update_layout(height=380)
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                routing_df=pd.DataFrame({
                    'Decision':rd.index,'Volume':rd.values,
                    'Cost':[ cost_map.get(d,avg_cost) for d in rd.index]})
                routing_df['Total_Cost']=routing_df['Volume']*routing_df['Cost']
                fig=px.bar(routing_df,x='Decision',y='Total_Cost',
                           color='Cost',color_continuous_scale='RdYlGn_r',
                           title="Cost by Routing Decision")
                fig.update_layout(height=380,coloraxis_showscale=False)
                st.plotly_chart(fig,use_container_width=True)
            total_before=len(intr)*avg_cost
            total_after=sum(routing_df['Total_Cost'])
            st.metric("Monthly Saving from Smart Routing",
                      f"EGP {(total_before-total_after):,.0f}",
                      f"↓ {(total_before-total_after)/total_before*100:.1f}%")

# ══════════════════════════════════════════════════════════════════
# PAGE — KPI PROJECTIONS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 KPI Projections":
    st.markdown(f"<h2 style='color:{NAVY};'>📈 KPI Baseline vs Post-Chatbot Projections</h2>",unsafe_allow_html=True)

    kpi_data=[
        ("Cost Per Interaction",f"EGP {avg_cost:.1f}","EGP 4.0",f"↓ {(avg_cost-4)/avg_cost*100:.0f}%",RED,GREEN),
        ("Avg Peak Wait Time",f"{peak_wait:.1f} min","< 1 min","↓ 88%",RED,GREEN),
        ("FCR Rate",f"{fcr_rate*100:.1f}%","77.5%",f"↑ {77.5-fcr_rate*100:.0f}pp",GOLD,GREEN),
        ("SLA Compliance",f"{sla_comp:.1f}%","87.5%",f"↑ {87.5-sla_comp:.0f}pp",GOLD,GREEN),
        ("Chatbot Deflection","0%",f"{defl_rate*100:.1f}%","✓ New KPI",GRAY,TEAL),
        ("Agent Overload Ratio",f"{overload:.2f}×","~1.05×","↓ Normalized",RED,GREEN),
        ("Monthly Service Cost",f"EGP {current_m/1e6:.1f}M",f"EGP {after_m/1e6:.1f}M",f"↓ EGP {monthly_sv/1e6:.1f}M",RED,GREEN),
        ("Annual Saving","—",f"EGP {annual_sv/1e6:.1f}M","✓ Net New",GRAY,GOLD),
    ]

    st.markdown("### 📊 KPI Comparison Table")
    kpi_df=pd.DataFrame(kpi_data,columns=['KPI','Before','After','Improvement','_bc','_ac'])
    st.dataframe(kpi_df[['KPI','Before','After','Improvement']],
                 use_container_width=True,hide_index=True)
    st.divider()

    c1,c2=st.columns(2)
    with c1:
        bv=[avg_cost,peak_wait,fcr_rate*100,sla_comp]
        av=[4.0,1.0,77.5,87.5]
        kn=['Cost/Int\n(EGP)','Peak Wait\n(min)','FCR Rate\n(%)','SLA\n(%)']
        fig=go.Figure()
        fig.add_trace(go.Bar(name='Before',x=kn,y=bv,marker_color=RED,opacity=0.85))
        fig.add_trace(go.Bar(name='After',x=kn,y=av,marker_color=GREEN,opacity=0.85))
        fig.update_layout(title="Core KPI: Before vs After",barmode='group',height=420)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","relative","total"],
            x=["Current Annual\nCost","Chatbot\nDeployment","Annual\nMaintenance","Annual\nSavings","Net Annual\nBenefit"],
            y=[current_m*12,-4000000,-500000,-annual_sv,0],
            connector={"line":{"color":GRAY}},
            decreasing={"marker":{"color":RED}},
            increasing={"marker":{"color":GREEN}},
            totals={"marker":{"color":TEAL}}))
        fig.update_layout(title="Annual ROI Waterfall",height=420)
        st.plotly_chart(fig,use_container_width=True)

    c1,c2,c3=st.columns(3)
    c1.metric("📉 Annual Saving",f"EGP {annual_sv/1e6:.1f}M","Projected")
    c2.metric("💹 Payback Period","< 18 months","On EGP 3–5M investment")
    c3.metric("🚀 Chatbot ROI",f"{annual_sv/4000000*100:.0f}%+","Year 1 return")

# ══════════════════════════════════════════════════════════════════
# PAGE — GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "🗺️ Geographic Analysis":
    st.markdown(f"<h2 style='color:{NAVY};'>🗺️ Geographic Analysis — NDB Branch & ATM Network</h2>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric("Total Branches",len(br))
    c2.metric("Total ATMs",len(atms))
    c3.metric("Governorates Covered",br['Governorate'].nunique())
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        br_v=br.dropna(subset=['Latitude','Longitude'])
        fig=px.scatter_mapbox(br_v,lat='Latitude',lon='Longitude',
            color='CSAT_Score',size='Avg_Daily_Footfall',
            hover_name='Branch_Name',
            hover_data=['Branch_Type','Governorate','Avg_Wait_Time_Minutes','FCR_Rate'],
            color_continuous_scale='RdYlGn',size_max=20,zoom=5,
            mapbox_style='carto-positron',
            title="Branch Network — CSAT Score & Footfall")
        fig.update_layout(height=480,coloraxis_colorbar_title="CSAT")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        gov=br.groupby('Governorate').agg(
            Count=('Branch_ID','count'),
            Avg_CSAT=('CSAT_Score','mean'),
            Avg_Wait=('Avg_Wait_Time_Minutes','mean')).reset_index().sort_values('Count',ascending=False)
        fig=px.bar(gov.head(15),x='Count',y='Governorate',orientation='h',
                   color='Avg_CSAT',color_continuous_scale='RdYlGn',
                   title="Branches by Governorate (Color=CSAT)")
        fig.update_layout(height=480,coloraxis_colorbar_title="CSAT")
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        bt=br.groupby('Branch_Type').agg(
            Avg_CSAT=('CSAT_Score','mean'),
            Avg_FCR=('FCR_Rate','mean'),
            Avg_Wait=('Avg_Wait_Time_Minutes','mean'),
            Count=('Branch_ID','count')).reset_index()
        fig=px.bar(bt,x='Branch_Type',y=['Avg_CSAT','Avg_FCR'],barmode='group',
                   title="Performance by Branch Type",
                   color_discrete_sequence=[TEAL,GOLD])
        fig.update_layout(height=340)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        at=atms['Location_Type'].value_counts()
        fig=px.pie(values=at.values,names=at.index,
                   title="ATM Distribution by Location Type",
                   color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=340)
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.markdown(f"<h2 style='color:{NAVY};'>👥 Customer Segmentation Analysis</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Retail",f"{(cust['Customer_Segment']=='Retail').sum():,}","70%")
    c2.metric("SME",f"{(cust['Customer_Segment']=='SME').sum():,}","~20%")
    c3.metric("Premium",f"{(cust['Customer_Segment']=='Premium').sum():,}","7%")
    c4.metric("Corporate",f"{(cust['Customer_Segment']=='Corporate').sum():,}","3%")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        seg_csat=cust.groupby('Customer_Segment')['CSAT_Score'].mean().reset_index()
        fig=px.bar(seg_csat,x='Customer_Segment',y='CSAT_Score',
                   color='Customer_Segment',
                   color_discrete_sequence=[NAVY,TEAL,GOLD,GREEN],
                   title="Avg CSAT by Segment")
        fig.update_layout(height=380,showlegend=False,yaxis_range=[0,5])
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        dig_seg=cust.groupby('Customer_Segment').agg(
            Digital=('Digital_User',lambda x:(x==True).mean()*100),
            Mobile=('Mobile_App_User',lambda x:(x==True).mean()*100),
            WhatsApp=('WhatsApp_Banking',lambda x:(x==True).mean()*100)).reset_index()
        fig=px.bar(dig_seg,x='Customer_Segment',y=['Digital','Mobile','WhatsApp'],
                   barmode='group',title="Digital Adoption by Segment (%)",
                   color_discrete_sequence=[NAVY,TEAL,GOLD])
        fig.update_layout(height=380)
        st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3=st.columns(3)
    with c1:
        fig=px.histogram(cust,x='Age',color='Customer_Segment',nbins=20,
                         opacity=0.7,title="Age Distribution by Segment",
                         color_discrete_sequence=[NAVY,TEAL,GOLD,GREEN])
        fig.update_layout(height=320)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        inc=cust.groupby(['Customer_Segment','Income_Bracket']).size().reset_index(name='count')
        fig=px.bar(inc,x='Customer_Segment',y='count',color='Income_Bracket',
                   title="Income Bracket by Segment",
                   color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=320)
        st.plotly_chart(fig,use_container_width=True)
    with c3:
        stat=cust.groupby(['Customer_Segment','Customer_Status']).size().reset_index(name='count')
        fig=px.bar(stat,x='Customer_Segment',y='count',color='Customer_Status',
                   color_discrete_map={'Active':GREEN,'Dormant':GOLD,
                                        'Suspended':RED,'Closed':GRAY},
                   title="Customer Status by Segment")
        fig.update_layout(height=320)
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — LOAN & CARD PORTFOLIO
# ══════════════════════════════════════════════════════════════════
elif page == "💳 Loan & Card Portfolio":
    st.markdown(f"<h2 style='color:{NAVY};'>💳 Loan & Card Portfolio Analysis</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Loan Portfolio",f"EGP {loans['Principal_Amount'].sum()/1e9:.2f}B")
    c2.metric("Outstanding Balance",f"EGP {loans['Outstanding_Balance'].sum()/1e9:.2f}B")
    c3.metric("Default Rate",f"{(loans['Loan_Status']=='Defaulted').mean()*100:.1f}%")
    c4.metric("Card Active Rate",f"{(cards['Card_Status']=='Active').mean()*100:.1f}%")
    st.divider()
    tab1,tab2=st.tabs(["🏦 Loans","💳 Cards"])
    with tab1:
        c1,c2,c3=st.columns(3)
        with c1:
            lt=loans['Loan_Type'].value_counts()
            fig=px.pie(values=lt.values,names=lt.index,title="Loan Type Distribution",hole=0.4)
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            ls=loans['Loan_Status'].value_counts()
            fig=px.bar(x=ls.index,y=ls.values,color=ls.index,
                       color_discrete_map={'Active':GREEN,'Closed':TEAL,
                                            'Defaulted':RED,'Under Review':GOLD,'Rejected':GRAY},
                       title="Loan Status Distribution")
            fig.update_layout(height=320,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        with c3:
            if 'Loan_Health_Score' in loans.columns:
                lh=loans['Loan_Health_Score'].value_counts()
                fig=px.pie(values=lh.values,names=lh.index,
                           color=lh.index,
                           color_discrete_map={'Healthy':GREEN,'At Risk':GOLD,'Defaulting':RED},
                           title="Loan Health Score",hole=0.4)
                fig.update_layout(height=320)
                st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(loans,x='Principal_Amount',nbins=30,
                             color_discrete_sequence=[TEAL],
                             title="Loan Amount Distribution")
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            miss=loans.groupby('Loan_Type')['Missed_Payments_Count'].mean().reset_index()
            fig=px.bar(miss,x='Loan_Type',y='Missed_Payments_Count',
                       color='Missed_Payments_Count',color_continuous_scale='Reds',
                       title="Avg Missed Payments by Loan Type")
            fig.update_layout(height=320,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
    with tab2:
        c1,c2,c3=st.columns(3)
        with c1:
            ct=cards['Card_Type'].value_counts()
            fig=px.pie(values=ct.values,names=ct.index,title="Card Type",hole=0.4,
                       color_discrete_sequence=[NAVY,TEAL,GOLD])
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            cs=cards['Card_Status'].value_counts()
            fig=px.bar(x=cs.index,y=cs.values,color=cs.index,
                       color_discrete_map={'Active':GREEN,'Blocked':RED,
                                            'Frozen':GOLD,'Expired':GRAY,
                                            'Lost':ORANGE,'Stolen':PURPLE},
                       title="Card Status Distribution")
            fig.update_layout(height=320,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        with c3:
            cn=cards['Card_Network'].value_counts()
            fig=px.pie(values=cn.values,names=cn.index,title="Card Network",hole=0.4,
                       color_discrete_sequence=[NAVY,TEAL,GOLD])
            fig.update_layout(height=320)
            st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — ERP & PROCESS FLOW
# ══════════════════════════════════════════════════════════════════
elif page == "⚙️ ERP & Process Flow":
    st.markdown(f"<h2 style='color:{NAVY};'>⚙️ ERP Integration & Digital Process Flow (BPM)</h2>",unsafe_allow_html=True)

    tab1,tab2,tab3=st.tabs(["🗄️ ERP Modules","🔄 BPM & Data Flow","📋 SDLC & RPA"])

    with tab1:
        st.subheader("Oracle ERP Modules Integrated with NDB Chatbot")
        erp_data={
            'ERP Module':['Oracle FLEXCUBE','Oracle CRM','Oracle Siebel',
                          'HR Module','BI & Reporting'],
            'Function':['Core Banking: Accounts, Balances, Loans, Cards',
                        'Customer Profiles, Segments, Interaction History',
                        'Complaint Management, SLA Tracking, Case Mgmt',
                        'Agent Profiles, Schedules, Queue Management',
                        'KPI Dashboard, Audit Logs, Analytics Engine'],
            'Chatbot Integration':['Real-time balance/loan/card API calls',
                                    'Single customer view before first response',
                                    'Auto complaint creation with reference number',
                                    'Smart queue routing by agent specialization',
                                    'All interactions logged for CBE compliance'],
            'Pain Point Addressed':['PP1,PP3,PP4','PP4,PP6','PP5','PP1,PP2','PP3,PP5,PP6']
        }
        st.dataframe(pd.DataFrame(erp_data),use_container_width=True,hide_index=True)

        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Sankey(
                node=dict(
                    pad=15,thickness=20,
                    label=["Customer","Web Chatbot","NLP Engine","Oracle FLEXCUBE",
                           "Oracle CRM","Oracle Siebel","BI Engine","Agent"],
                    color=[TEAL,NAVY,GOLD,GREEN,GREEN,GREEN,PURPLE,RED]),
                link=dict(
                    source=[0,1,2,2,2,2,1,6],
                    target=[1,2,3,4,5,6,7,7],
                    value=[100,100,30,20,15,35,35,35],
                    color=['rgba(0,123,138,0.3)']*8)))
            fig.update_layout(title="ERP Data Flow — Sankey Diagram",height=420)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            iso_data={
                'ISO 27001 Clause':['A.8.2','A.9.4','A.12.4','A.13.2','A.14.2','A.18.1'],
                'Title':['Information Classification','System Access Control',
                          'Logging & Monitoring','Information Transfer',
                          'Security in Development','Compliance with Legal Requirements'],
                'NDB Application':['Chatbot data classified Confidential — encrypted',
                                    'MFA mandatory before account data disclosed',
                                    'Full audit trail — tamper-proof CBE logs',
                                    'API calls encrypted via TLS 1.3',
                                    'Security testing in every SDLC phase',
                                    'CBE + Egyptian Data Protection Law mapped']}
            st.dataframe(pd.DataFrame(iso_data),use_container_width=True,hide_index=True)

    with tab2:
        st.subheader("Business Process Model — NDB AI Chatbot Interaction Flow")
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("1. Customer Input","Chat/Web","Query in AR/EN")
        c2.metric("2. NLP Engine","Intent Match","BERT-based NLU")
        c3.metric("3. ERP API","Oracle/SAP","Real-time data")
        c4.metric("4. Core Banking","Tx Update","Auto sync")
        c5.metric("5. Feedback","CSAT Log","BI Dashboard")

        st.markdown("""
        ```
        Customer → [Chatbot UI] → [NLP Engine] → [Intent Classifier]
                                                       ↓              ↓
                                               [ERP API Call]   [Human Escalation]
                                                       ↓              ↓
                                           [Oracle FLEXCUBE]    [Agent Queue]
                                                       ↓              ↓
                                           [Personalized Response]  [Context Transfer]
                                                       ↓
                                            [BI Interaction Log]
                                                       ↓
                                              [CSAT Collection]
        ```
        """)

        rpa_data={
            'RPA Process':['Complaint Auto-Logging','SLA Breach Alerts',
                           'Statement Generation','Card Block/Unblock',
                           'Loan Status Sync','Interaction Log Sync'],
            'Trigger':['Complaint filed via chatbot','Deadline approaching',
                       'Customer request','MFA verified','Schedule',
                       'After each interaction'],
            'ERP Target':['Oracle Siebel','Oracle Siebel','Oracle FLEXCUBE',
                          'Oracle FLEXCUBE','Oracle FLEXCUBE','BI & Reporting'],
            'Annual Saving (EGP)':['200,000','80,000','150,000',
                                    '120,000','90,000','100,000']}
        st.subheader("RPA Automation — NDB Back-Office Processes")
        st.dataframe(pd.DataFrame(rpa_data),use_container_width=True,hide_index=True)

    with tab3:
        st.subheader("Agile SDLC — 6-Phase NDB Chatbot Implementation")
        sdlc_data={
            'Phase':['Phase 1: Discovery','Phase 2: Design','Phase 3: Development',
                     'Phase 4: Testing','Phase 5: Deployment','Phase 6: Maintenance'],
            'Duration':['2 Weeks','3 Weeks','8 Weeks','3 Weeks','2 Weeks','Ongoing'],
            'Key Activities':['Pain point analysis, use case definition, CBE mapping',
                               'Architecture design, ERP integration plan, NLP selection',
                               'NLU model training, chatbot engine, ERP API integration',
                               'UAT, security testing, CBE audit trail validation',
                               'Web chat go-live, agent training, monitoring setup',
                               'Model retraining, performance monitoring, iteration'],
            'Deliverable':['Requirements document','Technical architecture',
                           'Working chatbot prototype','Test report + sign-off',
                           'Live production system','Monthly performance report']}
        st.dataframe(pd.DataFrame(sdlc_data),use_container_width=True,hide_index=True)

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:{GRAY};font-size:12px;'>
NDB — Nile Digital Bank | Analytics Dashboard |
</div>""", unsafe_allow_html=True)
