from atria_core.registry._module import Module
from atria_core.registry._module_config import ModuleConfig


class AnimalParams(ModuleConfig):
    name: str = "unknown"


class MammalParams(AnimalParams):
    legs: int = 4


class DogParams(MammalParams):
    breed: str = "labrador"


class SecialDogParams(MammalParams):
    breed: str = "labrador"


class AnimalModule[T: AnimalParams](
    Module[T],
):
    abstract = True

    def greet(self) -> str:
        return f"hi, I'm {self.config.name}"


class MammalModule[T: MammalParams](
    AnimalModule[T],
):
    abstract = True

    def describe(self) -> str:
        return f"{self.config.name} has {self.config.legs} legs"


class DogModule(MammalModule[DogParams]):
    def bark(self) -> str:
        return f"{self.config.breed} says woof"


class SpecialDog(MammalModule[SecialDogParams]):
    pass


dog = DogModule(DogParams(name="rex", breed="poodle"))
dog.greet()  # "hi, I'm rex"
dog.describe()  # "rex has 4 legs"
dog.bark()  # "poodle says woof"


special = SpecialDog()
special.config.breed  # "labrador" (default)

print("dog", dog._config_cls)
print("special", special._config_cls)
