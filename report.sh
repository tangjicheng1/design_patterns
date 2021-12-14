echo "代码行数统计:\n" > report.txt
cloc ./src >> report.txt
echo "\n圈复杂度统计:\n" >> report.txt
lizard >> report.txt