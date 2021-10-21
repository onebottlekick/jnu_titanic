import csv

def submit(id_num, answer):
    with open('submission.csv', 'w', newline='') as csvfile:
        fieldnames = ["PassengerId", "Survived"]
        answer_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        answer_writer.writeheader()
        for i, a in zip(id_num, answer):
            answer_writer.writerow({'PassengerId': i, 'Survived': a})  
