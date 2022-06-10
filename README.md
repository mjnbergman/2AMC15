# Continuous environment:

**Deze gaat kaolo hard**

Voer `main.py` uit en zie het verloop in de folder `./images`. Er wordt gewerkt aan een betere manier van visualiseren aan gebruikers maar `matplotlib` is bad.

# Status

- De environment heeft support voor meerdere robots (needs testing).
- De environment wordt volledig continuous bijgehouden met `Shapely` en gerenderd met `matplotlib` voor de robot.
- Death tiles werken ook al en killen robots die ze aanraken.


# Todo
- [ ] Overzetten in `gym` package.
- [ ] Reward definen (waarschijnlijk door area van goal tiles voor en na movement van elkaar af te trekken en te vermenigvuldigen met reward scalar).
- [ ] Battery drain met elke move, afhankelijk van bewogen afstand (ez).
- [ ] Level editor aansluiten op nieuw map formaat (zie `example-env.json`).
- [ ] Charging stations? (misschien niet handig met hoe battery % aan model wordt meegegeven?).
- [ ] Cleaned percentage loggen (orignele goal area - eind goal area)/originele goal area.
- [ ] Image wordt nu goed weergegeven aan model maar human-readable versie werkt nog niet goed.
- [ ] Track moves per robot.
- [ ] Documentatie.
