import streamlit as st
import sklearn

# <---------- Pages Setup ---------->
about_page1 = st.Page(
    page = "All-Pages/development.py",
    title = "Machine learning",
    icon = ":material/code:",
    default = True,
)

about_page2 = st.Page(
    page = "All-Pages/model_development.py",
    title = "Neural Network",
    icon = ":material/code:",
)

about_page3 = st.Page(
    page = "All-Pages/machine_learning_model.py",
    title = "Machine Learning Model",
    icon = ":material/terminal:",
)

about_page4 = st.Page(
    page = "All-Pages/neural_network_model.py",
    title = "Neural Network Model",
    icon = ":material/terminal:",
)


# <---------- Navigation Setup ---------->
#pg = st.navigation(pages=[about_page1, about_page2, about_page3, about_page4])


# <---------- Navigation Setup (with Section) ---------->
pg = st.navigation(
    {
        "Development": [about_page1,about_page2],
        "Models": [about_page3,about_page4],
    }
)


# <---------- Navigation Setup ---------->
pg.run()

