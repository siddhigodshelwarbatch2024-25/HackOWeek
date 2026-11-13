np.random.seed(42)

students = pd.DataFrame({
    "Student_ID": range(101, 121),
    "Name": [
        "Aarav", "Priya", "Rahul", "Sneha", "Aditya",
        "Ananya", "Rohan", "Kavya", "Vivek", "Neha",
        "Arjun", "Isha", "Karan", "Pooja", "Yash",
        "Simran", "Amit", "Riya", "Nikhil", "Meera"
    ],
    "Department": [
        "CSE", "CSE", "IT", "IT", "CSE",
        "ECE", "CSE", "IT", "ECE", "CSE",
        "IT", "CSE", "ECE", "IT", "CSE",
        "ECE", "IT", "CSE", "ECE", "IT"
    ],
    "Python": np.random.randint(55, 96, 20),
    "NumPy": np.random.randint(50, 96, 20),
    "Pandas": np.random.randint(55, 98, 20),
    "Attendance": np.random.randint(65, 100, 20)
})
