# %%
from dataclasses import dataclass

# %%


class Person:
    """person entity"""

    greeting = "Hi {0}. My name is {1}!"

    def __init__(self, name, surname, age):
        self.name = name
        self.surname = surname
        self.age = age

    def __repr__(self):
        return f"Person(name={self.name}, surname={self.surname}, age={self.age})"

    def say_hi(self, name):
        print(self.greeting.format(name, self.name))


# %%
P = Person("Pippi", "Longstocking", 11)
P

# %%
P.say_hi("Kalle")

# %%
P.name, P.surname = "Kalle", "Blomkvist"
P

# %%
data = [
    {"name": "Pippi", "surname": "Longstocking", "age": 11},
    {"name": "Kalle", "surname": "Blomkvist", "age": 10},
    {"name": "Karlsson-on-the-Roof", "surname": None, "age": 12},
]

characters = [Person(**row) for row in data]
for character in characters:
    character.say_hi("Reader")

# %%


class Animal:
    def __init__(self, age, diet):
        self.age = age
        self.diet = diet


# %%
Animal(1, "worms")

# %%


class Animal:
    def __init__(self, age, diet):
        self.age = age
        self.diet = diet

    def __repr__(self):
        return f"Animal(age={self.age}, diet='{self.diet}')"


# %%
Animal(1, "worms")

# %%


class School:
    def __init__(self, *fishes):
        self.fishes = list(fishes)


class Fish:
    def __add__(self, other):
        return School(self, other)


# %%
F1, F2 = Fish(), Fish()
F1 + F2

# %%
# path = Path('.').parent / 'data'
# path.absolute()

# %%


class Person:
    """
    person entity
    """

    def __init__(self, name, surname, age):
        self.name = name
        self.surname = surname
        self.age = age

    def __repr__(self):
        return f"Person(name={self.name}, surname={self.surname}, age={self.age})"

    def __lt__(self, other):
        return self.age < other.age


# %%
data = [
    {"name": "Pippi", "surname": "Longstocking", "age": 11},
    {"name": "Kalle", "surname": "Blomkvist", "age": 10},
    {"name": "Karlsson", "surname": "on-the-Roof", "age": 12},
]

characters = [Person(**row) for row in data]

# %%
sorted(characters)

# %%


class School:
    def __init__(self, *fishes):
        self.fishes = list(fishes)

    def __len__(self):
        return len(self.fishes)


# %%
S = School(Fish(), Fish())
len(S)

# %%


class School:
    def __init__(self, *fishes):
        self.fishes = list(fishes)

    def __getitem__(self, i):
        return self.fishes[i]


# %%
S = School(Fish(), Fish())
S[0]

# %%
S.__class__

# %%


class Fish:
    weight = 5
    color = "white"

    def __init__(self, w):
        self.weight = w


class ClownFish(Fish):
    color = "red"


# %%
c = ClownFish(w=15)
c.weight

# %%
c.color

# %%


class Mammal:
    produce = "milk"


class Dolphin(Fish, Mammal):
    pass


# %%
d = Dolphin(w=20)

# %%
d.produce

# %%
d.color

# %%


class Shark(Fish):
    def __init__(self, w=5000, teeth=121):
        self.teeth = teeth
        super().__init__(w=w)


S = Shark()
S.weight

# %%


@dataclass
class Person:
    name: str
    age: int


# %%
P1 = Person("Pippi", 11)
P2 = Person("Pippi", 11)
K = Person("Kalle", 10)

# %%
P1 == K

# %%
P1 == P2

# %%
