import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('student_data.csv')


def sex_and_total_score_bar_chart():
    df['total_score'] = df['G1'] + df['G2'] + df['G3']
    females_df = df.loc[df['sex'] == 'F']
    males_df = df.loc[df['sex'] == 'M']
    female_total_score = females_df['total_score'].sum()
    male_total_score = males_df['total_score'].sum()
    l1 = [male_total_score, female_total_score]
    l2 = ['Male', 'Female']
    plt.bar(l2, l1, width=0.4)
    plt.show()


def age_to_score_diagram():
    unique_age_values = sorted(df['age'].unique().tolist())
    print(unique_age_values)
    average_score_on_age = []
    df['total_score'] = df['G1'] + df['G2'] + df['G3']
    for i in unique_age_values:
        average = round(df.loc[df['age'] == i]['total_score'].sum() / len(df.loc[df['age'] == i]), 2)
        average_score_on_age.append(average)
    print(average_score_on_age)
    plt.plot(unique_age_values, average_score_on_age)
    plt.xlabel('Age')
    plt.ylabel('Total score')
    plt.yticks([i for i in range(0, 51, 5)])
    plt.show()


# x = np.linspace(-1.0, 1.0, 200)
# print(x)
# y = np.sqrt(1 - x**2)
# y1 = -np.sqrt(1 - x**2)
# plt.plot(x, y, 'b--')
# plt.plot(x, y1, 'b--')
# #plt.plot(x,y1)
# plt.show()
def generate_prime_numbers(lower_value, upper_value):
    l1 = []
    for number in range (lower_value, upper_value + 1):
        if number > 1:
            for i in range (2, number):
                if (number % i) == 0:
                    break
            else:
                l1.append(number)
    return l1


y = np.array(generate_prime_numbers(0, 1000))
x = np.array([i for i in range(len(y))])
devy = np.gradient(y)
plt.style.use('ggplot')
plt.plot(x, y, marker='.')
plt.plot(x, devy, marker='.')
for index in range(len(x)):
  plt.text(x[index], y[index], x[index], size=8)
  plt.text(x[index], devy[index], x[index], size=8)
plt.show()