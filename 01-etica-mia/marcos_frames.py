#Marcos (Frames):



class Frame:

    def __init__(self, name, parent=None, **slots):

        self.name = name

        self.parent = parent

        self.slots = slots



    def obtener(self, slot):

        if slot in self.slots:

            return self.slots[slot]

        if self.parent:

            return self.parent.obtener(slot)

        return None



animal = Frame("Animal", None, tiene_sangre=True)

perro = Frame("Perro", animal, sonido="ladrar")

firulais = Frame("Firulais", perro, color="café")



print("Firulais tiene sangre:", firulais.obtener("tiene_sangre"))

print("Sonido de Firulais:", firulais.obtener("sonido"))