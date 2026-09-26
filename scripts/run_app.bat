@echo off
title Assignment Duplicate Detector - NLP Mini Project
echo ====================================================================
echo        STARTING ASSIGNMENT DUPLICATE DETECTOR (STREAMLIT)
echo ====================================================================
echo.
cd /d %~dp0\..
python -m streamlit run app/app.py
pause
