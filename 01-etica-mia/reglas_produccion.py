#Sistema de Reglas de Producción

hechos = {"temperatura": 39, "tos": True, "dolor": False}



reglas = [

    (lambda h: h["temperatura"] > 38 and h["tos"], lambda h: h.update({"diagnóstico": "Fiebre con tos"})),

    (lambda h: h["temperatura"] > 38 and not h["tos"], lambda h: h.update({"diagnóstico": "Fiebre simple"})),

    (lambda h: h["dolor"], lambda h: h.update({"recomendación": "Analgésico"}))

]



for cond, accion in reglas:

    if cond(hechos):

        accion(hechos)



print(hechos)