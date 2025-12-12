from matplotlib.pyplot import boxplot, figure, grid, savefig, show, subplots, tight_layout, title, xlabel, xticks, ylabel, ylim
from numpy import array, arange, log, mean, random, sqrt
from scipy.stats import norm, sem

# Opening and reading current and target files.
Records, Targets, Skills, StudentIDs, ProblemsIDs, Gamma, Mean, sqrtVariance, Success, Failure, goldStus, boxPlots, barPlots, Simulation, sumScores, Counter, Width, DPI = open("student_sequences.txt").readlines(), open("gold_standard_sequences.txt").readlines(), open("problem_skill.txt").readlines(), range(1, 76), range(20), 0.5, -0.5, sqrt(0.1), 2, 1, {}, [], [], [], 0, 0, 0.1, 1200
Records, Targets, Skills, Betaqs = [list(map(int, Records[Record][:-1].split(","))) for Record in range(len(Records))], [list(map(int, Targets[Target][:-1].split(","))) for Target in range(len(Targets))], [int(Skills[Skill][:-1].split(",")[1]) for Skill in range(len(Skills))], [random.normal(Mean, sqrtVariance) for ProblemID in ProblemsIDs]
for StudentID in StudentIDs:
    goldStus[StudentID] = [Target[1:] for Target in Targets if Target[0] == StudentID]
maxTime = range(max([len(goldStus[Stu]) for Stu in goldStus]))

# Preparing averages for box plotting
for timeStep in maxTime:
    for Stu in goldStus:
        if timeStep <= len(goldStus[Stu]) - 1:
            sumScores += goldStus[Stu][timeStep][1]
            Counter += 1

    Simulation.append(sumScores / Counter)
    sumScores, Counter = 0, 0

boxPlots.append(Simulation)

# Preparing averages for bar plotting
ProblemIDs, Simulation = range(1, 21), []
for ProblemID in ProblemIDs:
    sumScores, Counter = 0, 0
    for Stu in goldStus:
        Activities = goldStus[Stu]
        for Activity in Activities:
            if Activity[0] == ProblemID:
                sumScores += Activity[1]
                Counter += 1
        
    Simulation.append(sumScores / Counter)

barPlots.append(Simulation)

# Mean Absolute Error (MAE) Error
def MAE(Students, Student):
    N = len(Students[Student])
    numberActivities = range(N)
    return sum([abs(Students[Student][Index][1] - goldStus[Student][Index][1]) for Index in numberActivities]) / N

# Mean Squared Error (MSE) Error
def MSE(Students, Student):
    N = len(Students[Student])
    numberActivities = range(N)
    return sum([(Students[Student][Index][1] - goldStus[Student][Index][1]) ** 2 for Index in numberActivities]) / N

def No_Skill_Learning(Betaq = -0.001, Lambda = 0.002, Rho = 0.001):
    GammaPrime, Students, Simulation, sumScores, Counter = log(1 / (1 - Gamma) - 1) - Betaq, {}, [], 0, 0
    for StudentID in StudentIDs:
        Students[StudentID] = [[Record[1], Record[3]] for Record in Records if Record[0] == StudentID]

    for Student in Students:
        # Activities for each s from S
        Activities, Problems = Students[Student], {}
        numberActivities = len(Activities) - 1
        for Activity in Activities:
            ProblemID = Activity[0]
            Problems[ProblemID] = [Activity[1] for Activity in Activities if Activity[0] == ProblemID]

        # Focusing on each q from Q
        ProblemsIDs = list(Problems.keys())
        for ProblemID in ProblemsIDs:
            # From t > 1
            Scores, Index, Counter, Rsq, Fsq = Problems[ProblemID], 1, 0, 0, 0
            Occurences = len(Scores) - 1
            # Calculating Rsq and Fsq regarding to the past for each q from Q
            while Index <= Occurences:
                Past = Scores[:Index]
                for ghat in Past:
                    if ghat == Success:
                        Rsq += 1
                    elif ghat == Failure:
                        Fsq += 1

                if Lambda * Rsq + Rho * Fsq > GammaPrime:
                    Scores[Index] = Success
                else:
                    Scores[Index] = Failure

                # Ready for the next time problem q begins being sovled
                Rsq, Fsq = 0, 0
                Index += 1

            # Submitting calculated scores
            Index = 0
            while Index <= numberActivities:
                if Activities[Index][0] == ProblemID:
                    Activities[Index] = [ProblemID, Scores[Counter]]
                    Counter += 1
                Index += 1

        # Updating Records for each s from S
        Students[Student] = Activities

    # Preparing averages for box plotting
    for timeStep in maxTime:
        for Student in Students:
            if timeStep <= len(Students[Student]) - 1:
                sumScores += Students[Student][timeStep][1]
                Counter += 1

        Simulation.append(sumScores / Counter)
        sumScores, Counter = 0, 0

    boxPlots.append(Simulation)

    # Preparing averages for bar plotting
    ProblemIDs, Simulation = range(1, 21), []
    for ProblemID in ProblemIDs:
        sumScores, Counter = 0, 0
        for Student in Students:
            Activities = Students[Student]
            for Activity in Activities:
                if Activity[0] == ProblemID:
                    sumScores += Activity[1]
                    Counter += 1
        
        Simulation.append(sumScores / Counter)

    barPlots.append(Simulation)
    return sum([MAE(Students, Student) for Student in Students]), sum([MSE(Students, Student) for Student in Students])

def Heterogeneous_Learning(Lambda = 0.2, Rho = 0.05):
    GammaPrime, Students, Simulation, sumScores, Counter = log(1 / (1 - Gamma) - 1), {}, [], 0, 0
    for StudentID in StudentIDs:
        Students[StudentID] = [[Record[1], Record[3]] for Record in Records if Record[0] == StudentID]

    for Student in Students:
        # Activities for each s from S
        Activities, Problems = Students[Student], {}
        numberActivities = len(Activities) - 1
        for Activity in Activities:
            ProblemID = Activity[0]
            Problems[ProblemID] = [Activity[1] for Activity in Activities if Activity[0] == ProblemID]

        # Focusing on each q from Q
        ProblemsIDs = list(Problems.keys())
        for ProblemID in ProblemsIDs:
            # From t > 1
            Scores, GammaPrime, Index, Counter, Rsq, Fsq = Problems[ProblemID], GammaPrime - Betaqs[ProblemID - 1], 1, 0, 0, 0
            Occurences = len(Scores) - 1
            # Calculating Rsq and Fsq regarding to the past for each q from Q
            while Index <= Occurences:
                Past = Scores[:Index]
                for ghat in Past:
                    if ghat == Success:
                        Rsq += 1
                    elif ghat == Failure:
                        Fsq += 1

                if Lambda * Rsq + Rho * Fsq > GammaPrime:
                    Scores[Index] = Success
                else:
                    Scores[Index] = Failure

                # Ready for the next time problem q begins being sovled
                Rsq, Fsq = 0, 0
                Index += 1

            # Submitting calculated scores
            Index = 0
            while Index <= numberActivities:
                if Activities[Index][0] == ProblemID:
                    Activities[Index] = [ProblemID, Scores[Counter]]
                    Counter += 1
                Index += 1

        # Updating Records for each s from S
        Students[Student] = Activities

    # Preparing averages for box plotting
    for timeStep in maxTime:
        for Student in Students:
            if timeStep <= len(Students[Student]) - 1:
                sumScores += Students[Student][timeStep][1]
                Counter += 1

        Simulation.append(sumScores / Counter)
        sumScores, Counter = 0, 0

    boxPlots.append(Simulation)

    # Preparing averages for bar plotting
    ProblemIDs, Simulation = range(1, 21), []
    for ProblemID in ProblemIDs:
        sumScores, Counter = 0, 0
        for Student in Students:
            Activities = Students[Student]
            for Activity in Activities:
                if Activity[0] == ProblemID:
                    sumScores += Activity[1]
                    Counter += 1
        
        Simulation.append(sumScores / Counter)

    barPlots.append(Simulation)
    return sum([MAE(Students, Student) for Student in Students]), sum([MSE(Students, Student) for Student in Students])

def One_Skill_Learning(Betaq = -0.001, Lambda = 0.002, Rho = 0.001):
    GammaPrime, Students, Simulation, sumScores, Counter = log(1 / (1 - Gamma) - 1) - Betaq, {}, [], 0, 0
    for StudentID in StudentIDs:
        Students[StudentID] = [[Record[1], Record[3]] for Record in Records if Record[0] == StudentID]

    for Student in Students:
        # From t > 1
        Activities, Index, Rsk, Fsk = Students[Student], 1, 0, 0
        numberActivities = len(Activities) - 1
        while Index <= numberActivities:
            Problem, Past = Activities[Index][0], Activities[:Index]
            for Activity in Past:
                if Skills[Activity[0] - 1] == Skills[Problem - 1]:
                    if Activity[1] == Success:
                        Rsk += 1
                    elif Activity[1] == Failure:
                        Fsk += 1
                        
            # As problems and skills have a functional relation, just one time Iqk equals 1 for each q from Q. So:
            if Lambda * Rsk + Rho * Fsk > GammaPrime:
                Activities[Index][1] = Success
            else:
                Activities[Index][1] = Failure

            # Ready for the next time problem q begins being sovled
            Rsk, Fsk = 0, 0
            Index += 1

        # Updating Records for each s from S
        Students[Student] = Activities

    # Preparing averages for box plotting
    for timeStep in maxTime:
        for Student in Students:
            if timeStep <= len(Students[Student]) - 1:
                sumScores += Students[Student][timeStep][1]
                Counter += 1

        Simulation.append(sumScores / Counter)
        sumScores, Counter = 0, 0

    boxPlots.append(Simulation)

    # Preparing averages for bar plotting
    ProblemIDs, Simulation = range(1, 21), []
    for ProblemID in ProblemIDs:
        sumScores, Counter = 0, 0
        for Student in Students:
            Activities = Students[Student]
            for Activity in Activities:
                if Activity[0] == ProblemID:
                    sumScores += Activity[1]
                    Counter += 1
        
        Simulation.append(sumScores / Counter)

    barPlots.append(Simulation)
    return sum([MAE(Students, Student) for Student in Students]), sum([MSE(Students, Student) for Student in Students])

# Low Learning, No-Skill PFA
print(No_Skill_Learning())

# High Learning, No-Skill PFA
print(No_Skill_Learning(-0.01, 0.8, 0.2))

# Heterogeneous Learning, No-Skill PFA
print(Heterogeneous_Learning())

# Low Learning, One-Skill PFA
print(One_Skill_Learning())

# High Learning, One-Skill PFA
print(One_Skill_Learning(-0.01, 0.8, 0.2))

# Box Plotting
figure(figsize = (15, 10))
boxplot(boxPlots, notch = True, patch_artist = True, boxprops = dict(facecolor = "lightblue", color = "blue"), medianprops = dict(color = "red"), whis = 1.5)

# Creating an extended figure for better visibility of outliers
xticks([1, 2, 3, 4, 5, 6], ["Gold Standard", "No-Skill Low", "No-Skill High", "Heterogeneous", "One-Skill Low", "One-Skill High"])
xlabel("Time Steps")
ylabel("The average of all simulated student scores on all problems in time")
title("Box Plot of Simulation")
grid(True)
savefig("Box Plots", dpi = DPI)
show()

# Creating a single figure for all bar plots
Figure, ax = subplots(figsize = (15, 7))

# Plotting each bar plot
for Index, Simulation in enumerate(barPlots):
    Means = mean(Simulation)
    # 95% Confidence Interval
    Ci = norm.interval(0.95, loc = Means, scale = sem(Simulation)) 
    # x positions based on actual length of scores
    ax.bar(arange(len(Simulation)) + Index * Width, Simulation, Width, yerr = array([[Means - Ci[0]], [Ci[1] - Means]]), label = f"Simulation {Index + 1}", capsize = 5)

# Customizing the plot
ax.set_xlabel("Problem IDs")
ax.set_ylabel("Average student scores over all time-steps for each question for each simulated setting")
ax.set_title("Bar Plot for Each Simulation Across 20 Questions")
ax.set_xticks(arange(20))
ax.set_xticklabels([f"Q{ProblemID}" for ProblemID in range(1, 21)])
ax.legend()
tight_layout()
savefig("Bar Plots", dpi = DPI)
show()
