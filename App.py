import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from google import genai

st.set_page_config(page_title="AI Credit Risk System", layout="wide")

# ---------- Custom UI ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.block-container {
    padding-top: 2rem;
}
.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
}
.sub-title {
    text-align: center;
    color: #f0f0f0;
    margin-bottom: 30px;
}
.card {
    background: rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}
.footer {
    text-align: center;
    margin-top: 40px;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEYS")

model = joblib.load("extra_trees_credit_model.pkl")

encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

st.markdown("<div class='main-title'> AI Credit Risk Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning + CIBIL Score + AI Explanation</div>", unsafe_allow_html=True)

# ---------- Input Section ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 80, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.number_input("Job (0-3)", 0, 3, 1)
    housing = st.selectbox("Housing", ["own", "rent", "free"])

with col2:
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "quite rich"])
    credit_amount = st.number_input("Credit Amount", 0, 100000, 5000)
    duration = st.number_input("Duration (months)", 1, 60, 12)

cibil_score = st.slider("CIBIL Score", 300, 900, 750)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Prepare Data ----------
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# ---------- AI Function ----------
def get_ai_explanation(cibil, credit_amt, duration_months, prediction):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        risk_label = "Low Risk" if prediction == 1 else "High Risk"

        prompt = f"""
        You are a very experienced Credit Risk Manager working in a top Indian bank.
        Applicant profile:
        - CIBIL Score: {cibil}
        - Requested Credit Amount: ₹{credit_amt:,}
        - Loan Duration: {duration_months} months
        - Machine Learning Model Prediction: {risk_label}

        Write a very professional, realistic, clear and actionable
        credit risk summary + strong recommendations.

        Use simple but professional language that a customer can understand.
        Be strict & honest when score is low.
        Be encouraging & give clear next steps when improvement is possible.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        if cibil < 550:
            return (
                "**LOAN REJECTED — VERY HIGH RISK**\n\n"
                "Your CIBIL score is currently in the **critical zone** (below 550).\n"
                "At this level almost all banks and NBFCs **reject** the application.\n\n"

                "**Most Important Things You Should Do (Next 6–24 months):**\n"
                "• Immediately clear / settle **all overdue** and **written-off** accounts\n"
                "• Take NOC (No Objection Certificate) from every lender after full settlement\n"
                "• Close all unnecessary / very old inactive accounts if they have negative remarks\n"
                "• **Do NOT** apply for any new loan / credit card for minimum 6–9 months\n"
                "• Start with a **secured credit card** (very small limit) and use it responsibly\n"
                "• Keep credit utilisation **below 10–15%** when you get any credit\n"
                "• Make every single payment **5–7 days before due date**\n"
                "• Check your CIBIL report every month (www.cibil.com)\n"
                "• Dispute every error you find (very important!)\n\n"

                "**Realistic Target:** Try to reach **at least 650–680** before applying again\n"
                "With serious discipline → visible improvement in **12–24 months** is possible"
            )

        elif cibil < 650:
            return (
                "**HIGH RISK — Very Difficult to Get Approval**\n\n"
                "Current CIBIL score is low.\n"
                "Most banks will either **reject** or give very **high interest rate + low amount**.\n\n"

                "**Best Action Plan Right Now (3–12 months):**\n"
                "• Clear **all** past due / overdue EMI and credit card dues **urgently**\n"
                "• Reduce **credit utilisation** to **below 25–30%** (ideally below 20%)\n"
                "• **No new applications** for loans or credit cards for at least 6 months\n"
                "• Pay every EMI / credit card bill **very early** (5–10 days before due date)\n"
                "• Consider a **small secured credit card** → use it lightly & pay in full\n"
                "• If possible → become **authorized user** on any family member's good card\n"
                "• Keep checking CIBIL report → **fix every single mistake** you find\n"
                "• Target **680–720+** → chances improve **dramatically**\n\n"

                "Small correct steps → very fast improvement in score"
            )

        elif cibil < 750:
            return (
                "**MODERATE RISK — Approval Possible but with Conditions**\n\n"
                "Your score is **average/fair**.\n"
                "Chances of approval → **Yes** but mostly with:\n"
                "• Higher interest rate\n"
                "• Lower loan amount\n"
                "• Sometimes extra security / guarantor\n\n"

                "**How to Improve → Get Much Better Terms:**\n"
                "• Bring **credit utilisation** down below **25–30%**\n"
                "• Maintain **100% on-time payment** for next **6–12 months** (very powerful)\n"
                "• **Avoid** too many loan / card applications in short time\n"
                "• Pay off **smallest balances first** (snowball method — feels good)\n"
                "• Keep your **oldest credit accounts open** (helps credit history length)\n"
                "• If you have high-interest credit card debt → consider **balance transfer**\n"
                "• Regularly check CIBIL → aim to cross **750–780** as fast as possible\n\n"

                "You are very close to the **prime segment** — just **consistent behaviour** needed"
            )

        else:  # 750+
            return (
                "**LOW RISK — Very Good Credit Profile ✓**\n\n"
                "Excellent CIBIL score + good model prediction → **strong chance of approval**\n\n"

                "**How to Maintain / Get Even Better Offers:**\n"
                "• Continue **100% on-time payments** — this is your biggest strength\n"
                "• Try to keep **credit utilisation below 20–30%**\n"
                "• **Avoid unnecessary** hard inquiries (multiple applications)\n"
                "• Take benefit of **pre-approved offers** from your existing banks\n"
                "• Keep old accounts open → longer credit history = better score\n"
                "• Build **emergency fund** → so you never miss EMI even in tough months\n"
                "• Check your credit report once a year (just to be safe)\n\n"

                "You are already in a **very strong position** — protect it!"
            )

# ---------- Prediction & Output ----------
if st.button("🚀 Analyze Credit Risk", use_container_width=True):

    # Always run ML prediction (we'll use it in explanation too)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ────────────────────────────────
    #  Decision logic
    # ────────────────────────────────
    if cibil_score < 550:
        st.error("❌ Loan Rejected - Very Low CIBIL Score (< 550)")
        # Still show probability & explanation
        st.markdown("### 📊 Model Risk Score (for reference)")
        st.progress(int(probability * 100))
        st.write(f"Model Confidence (low-risk probability): {round(probability * 100, 2)}%")

    else:
        if prediction == 1 and cibil_score >= 700:
            st.success("✅ Loan Approved (Low Risk)")
        elif prediction == 1 and 600 <= cibil_score < 700:
            st.warning("⚠ Manual Review Required")
        else:
            st.error("❌ Loan Rejected (High Risk according to model)")

        st.markdown("### 📊 Approval Probability")
        st.progress(int(probability * 100))
        st.write(f"Confidence Score: {round(probability * 100, 2)}%")

    # ────────────────────────────────
    #  Always show AI explanation
    # ────────────────────────────────
    st.markdown("### 🤖 AI Risk Analysis & Recommendations")
    explanation = get_ai_explanation(cibil_score, credit_amount, duration, prediction)
    st.markdown(explanation)

st.markdown("<div class='footer'>© 2026 AI Credit Risk System | Streamlit + ML + Gemini</div>", unsafe_allow_html=True)