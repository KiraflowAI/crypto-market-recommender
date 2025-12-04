import subprocess

print("🔧 Aktualisiere requirements.txt ...")

# requirements.txt automatisch neu erzeugen
with open("requirements.txt", "w") as f:
    output = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    f.write(output)

print("✅ Fertig! requirements.txt wurde erfolgreich aktualisiert.")
