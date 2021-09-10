with open('labels.txt') as labels:
    target_dataset = [1 if lbl[0] == 'p' else 0 for lbl in labels.readlines()]
print(len(target_dataset))