---
title: "Text Adventure Game Development"
date: 2023-12-03
tags: [Python, Passion Project, game]
header:
  image: "/images/squid-game.png"
excerpt: "(Python - Machine Learning) At Apprentice Chef, a new subscription option, Halfway There, was just launched to a sample group of customers. Customers can receive a half bottle of wine from a local California vineyard every Wednesday. When promoting this service to a wider audience, we want to know which customers will subscribe to Halfway There. Based on a fictitious business case: Apprentice Chef, Inc. built by Professor Chase Kusterer from Hult International Business School"
mathjax: "true"
toc: true
toc_label : "Navigate"
---

## Text Adventure Game - Squid Game
By: Jorge Solis<br>
Hult International Business School<br>
<br>
<br>
Jupyter notebook and dataset for this analysis can be found here: [Portfolio-Projects](https://github.com/jorgesolisservelion/portfolio) 
<br>
<br>

***

~~~

███████╗ ██████╗ ██╗   ██╗██╗██████╗      ██████╗  █████╗ ███╗   ███╗███████╗
██╔════╝██╔═══██╗██║   ██║██║██╔══██╗    ██╔════╝ ██╔══██╗████╗ ████║██╔════╝
███████╗██║   ██║██║   ██║██║██║  ██║    ██║  ███╗███████║██╔████╔██║█████╗  
╚════██║██║▄▄ ██║██║   ██║██║██║  ██║    ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  
███████║╚██████╔╝╚██████╔╝██║██████╔╝    ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗
╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═════╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
                                                                             
~~~

Welcome to the Squid Game, where you can win a million dollars if you survive until the end of this game. The rules are simple; you just have to survive. Next, you will go through three stages of the game. The first one is called "Red Light, Green Light", the second is called the "Honeycomb Challenge", and the third is called "Tug of War". Be careful, and may the odds be in your favor.

Just a reminder:
* stage_1 : "Red Light, Green Light"
* stage_2 : "Honeycomb Challenge"
* stage_3 : "Tug of War"


Siren Pictures Inc., & Hwang Dong-hyuk. (2016). Squid Game. Retrieved from https://www.netflix.com

```python

# Import libraries
import time
import random 

# Function to print text with a delay
def print_with_delay(text, delay=1):
    print(text)
    time.sleep(delay)

# Function to handle user input and ensure valid choices
def get_user_input(prompt, choices):
    while True:
        user_input = input(prompt).lower()
        if user_input in choices:
            return user_input
        else:
            print_with_delay("Invalid choice. Please try again.")

# Introduction and game setup
def introduction():
    border = "*" * 60
    message = "Welcome to the Squid Game!"

    print(border)
    print("*" + " " * 58 + "*")
    print(f"*{message:^58s}*") 
    print("*" + " " * 58 + "*")
    print(border)
    print_with_delay("\nYou chose to be in this mysterious game, where life is at stake.")
    print_with_delay("The only way out of here is by winning each challenge, surviving, and winning the game.")
    print_with_delay("""We know you have debts, and you can't pay them. 
    That's why you're here of your own free will. 
    If you win the game, you'll have a million dollars that will save you from your financial misfortune. 
    But if you lose, you'll also lose your life.""")
    print_with_delay("\nLet's begin.\n")

# Stage 1: Red Light, Green Light
def stage_one():
    print_with_delay("Stage 1: Red Light, Green Light")
    print_with_delay("\nYou enter a vast playground with a giant doll at the end.🪆")
    print_with_delay("The rules are simple: move on Green Light, stop on Red Light.")
    print_with_delay("You have to move to position number 3 to advance to the next stage of the game. You need to decide whether to move or not each time you are asked.\n")
    print_with_delay("If it's a green light 🟢 and you chose to move, you'll move one position forward.")
    print_with_delay("But if it's a red light 🔴 and you decided to move, you'll go back to position 1.\n")
    print_with_delay("You have to guess if the light is green or red and decide whether to move or not.\n")
    
    # Variables for the game
    player_position = 0
    max_position = 2
    timer = 0
    is_green_light = False

    while player_position < max_position:
        user_choice = get_user_input("Do you want to move? (yes/no): ", ["yes", "no"])
        timer+=1
        if user_choice == "yes":
            if is_green_light:
                player_position += 1
                print_with_delay("\n Green Light! 🟢 You move one step forward.\n")
                print_with_delay(f"You are in the position N°{player_position+1}")
            else:
                print_with_delay("\n Red Light!🔴 You must stop.\n")
                player_position=0
                print_with_delay(f"You return to position N°{player_position+1}")
        else:
            print_with_delay("You chose not to move.")
            if is_green_light == True:
                print_with_delay("\n It was Green Light 🟢\n")
            else:
                print_with_delay("\n It was Red Light 🔴 \n")

        # Generate a random event for Green Light
        is_green_light = random.choice([True, False])
        if timer == 10 or timer == 15 or timer ==20:
            bored = get_user_input("What happened? Can't you proceed? Do you want to withdraw from the game? [yes/no]:", ["yes", "no"])
            if bored == "yes":
                break
            
    # Check if the player reached the end
    if player_position == max_position:
        print_with_delay("Congratulations! You successfully crossed the playground.")
        print_with_delay("""You're getting closer each time to saving yourself from 
        your financial misfortune and being able to pay off your debts.""")
        print_with_delay("Now, It's time to go to the next stage.\n")
        stage_two()
    else:
        print_with_delay("Game over. You didn't reach the end in time.")
        fail()

# Stage 2: Honeycomb Challenge
def stage_two():
    print_with_delay("Stage 2: Honeycomb Challenge")
    print_with_delay("\nIn this challenge, you must carve out a specific shape from a honeycomb.")

    # Define the allowed shapes
    allowed_shapes = ["circle", "square", "triangle"]

    # Choose a random shape for the challenge
    target_shape = random.choice(allowed_shapes)

    print_with_delay(f"The target shape is a {target_shape.capitalize()}.")
    print_with_delay("You have to solve this math problem to move to the next stage of the game")
    
    if target_shape == "circle":
        user_solution = problem_math_stage_two_circle()
        if user_solution == True:
            print_with_delay("Congratulations! You successfully carved the correct shape.\n")
            print_with_delay("""You are one step away from glory. 
            You will be able to pay off your debts and be the pride of your family. 
            But be careful, if you lose, you won't have time to tell what happened here.""")
            stage_three()
        else:
            print_with_delay("Oops! Your carved shape is incorrect.")
            fail()
    elif target_shape == "square":
        user_solution = problem_math_stage_two_square()
        if user_solution == True:
            print_with_delay("Congratulations! You successfully carved the correct shape.\n")
            print_with_delay("""You are one step away from glory. 
            You will be able to pay off your debts and be the pride of your family. 
            But be careful, if you lose, you won't have time to tell what happened here.""")
            stage_three()
        else:
            print_with_delay("Oops! Your carved shape is incorrect.")
            fail()
    elif target_shape == "triangle":
        user_solution = problem_math_stage_two_triangle()
        if user_solution == True:
            print_with_delay("Congratulations! You successfully carved the correct shape.\n")
            print_with_delay("""You are one step away from glory. 
            You will be able to pay off your debts and be the pride of your family. 
            But be careful, if you lose, you won't have time to tell what happened here.""")
            stage_three()
        else:
            print_with_delay("Oops! Your carved shape is incorrect.")
            fail()
    else:
        print_with_delay("There was something wrong")
        fail()
    
   
def problem_math_stage_two_circle():
    num1 = random.randint(1,10)
    num2 = random.randint(1,10)
    operation = "+"
    solve = f"{num1} {operation} {num2}"
    answer = eval(solve)
    
    while True:
        user_answer = input(prompt=f"\n Solve this math problem: {solve} \n")
        try:
            choice = int(user_answer)
        except:
            input(prompt = "Write a number in digits. Press enter to try again.\n")
            continue
        if choice == answer:
            print_with_delay("\n The answer is correct!! 👍 ✅ \n")
            return True
        else:
            print_with_delay("\n That is not the answer. Sorry 😢 \n")
            return False
        
def problem_math_stage_two_square():
    num1 = random.randint(1,10)
    num2 = random.randint(1,10)
    operation = "-"
    solve = f"{num1} {operation} {num2}"
    answer = eval(solve)
    
    while True:
        user_answer = input(prompt=f"\n Solve this math problem: {solve} \n")
        try:
            choice = int(user_answer)
        except:
            input(prompt = "Write a number in digits. Press enter to try again.\n")
            continue
        if choice == answer:
            print_with_delay("\n The answer is correct!! 👍 ✅ \n")
            return True
        else:
            print_with_delay("\n That is not the answer. Sorry 😢 \n")
            return False
    
def problem_math_stage_two_triangle():
    num1 = random.randint(1,10)
    num2 = random.randint(1,10)
    operation = "*"
    solve = f"{num1} {operation} {num2}"
    answer = eval(solve)
    
    while True:
        user_answer = input(prompt=f"\n Solve this math problem: {solve} \n")
        try:
            choice = int(user_answer)
        except:
            input(prompt = "Write a number in digits. Press enter to try again.\n")
            continue
        if choice == answer:
            print_with_delay("\n The answer is correct!! 👍 ✅ \n")
            return True
        else:
            print_with_delay("\n That is not the answer. Sorry 😢 \n")
            return False
    
# Stage 3: Tug of War
def stage_three():
    print_with_delay("\nStage 3: Tug of War\n")
    print_with_delay("You find yourself in a team-based challenge: Tug of War.")
    print_with_delay("To win this game, you have to be stronger than the other team.")
    print_with_delay("To earn more strength you can guess the riddles.")
    print_with_delay("If you answer correctly, you will gain more strength.")
    print_with_delay("Otherwise, you will compete against random forces, and luck will determine the winner of the Squid Game.")
    team_a=0
    team_b=0
    for x in range(3):
        team_a += calculate_team_strength()
        team_b += random.randint(1, 8)
    
    print_with_delay(f"Your team has a strength of {team_a}.")
    print_with_delay(f"Your rivals has a strength of {team_b}.")
    
    # Determine the winner of the Tug of War
    if team_a > team_b:
        print_with_delay("Your team wins the Tug of War! You successfully pull the other team.")
        win()
    elif team_a< team_b:
        print_with_delay("Your rivals wins the Tug of War! Unfortunately, your team couldn't pull hard enough.")
        fail()
    else:
        print_with_delay("It's a draw! The Tug of War ends without a clear winner.")
        fail()

def calculate_team_strength():
    print_with_delay("\n To determine your team's strength, solve the following riddle:")
    
    # Defining a list of riddles, their answers, and hints
    riddles = [
        {"question": "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?", "answer": "an echo", "hint": "Think about sounds bouncing back."},
        {"question": "The more you take, the more you leave behind. What am I?", "answer": "footsteps", "hint": "Consider what is left behind when you walk."},
        {"question": "What has keys but can't open locks?", "answer": "a piano", "hint": "It produces music."},
        {"question": "I'm tall when I'm young, and short when I'm old. What am I?", "answer": "a candle", "hint": "Think about how a candle burns."},
        {"question": "What comes once in a minute, twice in a moment, but never in a thousand years?", "answer": "the letter 'M'", "hint": "Think about time and the alphabet."},
        {"question": "The more you have of it, the less you see. What is it?", "answer": "darkness", "hint": "Consider the absence of light."},
        {"question": "What has a heart that doesn't beat?", "answer": "an artichoke", "hint": "Think about vegetables."},
        {"question": "I have keys but open no locks. I have space but no room. You can enter, but you can't go inside. What am I?", "answer": "a keyboard", "hint": "Think about computer accessories."},
        {"question": "I'm not alive, but I can grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?", "answer": "fire", "hint": "Think about elements."},
        {"question": "The more you take, the more you leave behind. What am I?", "answer": "footsteps", "hint": "Consider what is left behind when you walk."},
    ]
    
    # Choosing a random riddle
    riddle = random.choice(riddles)
    print_with_delay("\n"+riddle["question"])
    print_with_delay("\n Hint: " + riddle["hint"]+"\n")
    user_answer = input(prompt="Your answer: ").lower()

    if user_answer == riddle["answer"]:
        print_with_delay("Correct! Your team gains strength.\n")
        return random.randint(10, 20)
    else:
        print_with_delay("\n Incorrect!. Your team's strength is affected.\n")
        print_with_delay("The answer is: " + riddle["answer"]+"\n")
        return random.randint(5, 10)    
        
# Win function
def win():
    print_with_delay("Congratulations! You have successfully won the Squid Game.")
    print("""
    
    
 $$\     $$\                         $$\      $$\ $$\           
\$$\   $$  |                        $$ | $\  $$ |\__|          
 \$$\ $$  /$$$$$$\  $$\   $$\       $$ |$$$\ $$ |$$\ $$$$$$$\  
  \$$$$  /$$  __$$\ $$ |  $$ |      $$ $$ $$\$$ |$$ |$$  __$$\ 
   \$$  / $$ /  $$ |$$ |  $$ |      $$$$  _$$$$ |$$ |$$ |  $$ |
    $$ |  $$ |  $$ |$$ |  $$ |      $$$  / \$$$ |$$ |$$ |  $$ |
    $$ |  \$$$$$$  |\$$$$$$  |      $$  /   \$$ |$$ |$$ |  $$ |
    \__|   \______/  \______/       \__/     \__|\__|\__|  \__|
                                                               
                                                               
                                                               

    """)
    print("💸💸💸💸💸💸💸 1MM$$$ 💵💵💵💵💵💵💵💵💵💵")

# Fail function
def fail():
    print_with_delay("Unfortunately, you did not survive the challenge. Game over.")

# Main game function
def play_squid_game():
    introduction()
    # Start the game with Stage 1
    stage_one()
    
play_squid_game()

```

