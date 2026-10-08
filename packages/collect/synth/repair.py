"""Mise aux volumes exigés par le guide (L1.0), de façon déterministe.

Cibles par journée exportée : 60 à 250 mails, 4 à 10 événements, 3 à 12 tâches,
5 à 30 fichiers. On génère d'abord un monde « naturel », puis on corrige localement :
- mails     : ajout/retrait de notifications automatiques (jamais reportées → effet local) ;
- événements: ajout de réunions ordinaires / retrait de réunions de remplissage ;
- tâches    : le manager rattrape ses plus vieux retards / petites tâches du jour ;
- fichiers  : exports automatiques de la nuit / retrait de fichiers de bruit.
Les objets porteurs d'une affaire ou d'un piège ne sont jamais retirés.
"""

from __future__ import annotations

from datetime import timedelta

from collect.synth.clock import PARIS, is_working_day
from collect.synth.window import snapshot

LIMITS = {"mail": (60, 250), "calendar": (4, 10), "task": (3, 12), "file": (5, 30)}


class RepairError(RuntimeError):
    pass


def _removable(obj) -> bool:
    return obj.case_id is None and not (obj.tags - {"notification", "newsletter", "malformed_html"})


def repair(world, max_rounds: int = 40) -> int:
    """Corrige jusqu'à stabilité. Renvoie le nombre de passes qui ont dû corriger quelque chose."""
    for rounds in range(max_rounds):
        changed = False
        for day in world.export_days:
            changed |= _fix_mails(world, day)
            changed |= _fix_events(world, day)
            changed |= _fix_tasks(world, day)
            changed |= _fix_files(world, day)
        if not changed:
            return rounds
    raise RepairError("les volumes ne se stabilisent pas : revoir les paramètres du générateur")


def _fix_mails(world, day) -> bool:
    low, high = LIMITS["mail"]
    snap = snapshot(world, day)
    changed = False
    n = len(snap.mails)
    while n < low:
        world.add_filler_mail(day - timedelta(days=1))
        n += 1
        changed = True
    if n > high:
        last_day_only = [m for m in snap.mails if m.sent_at > snap.now - timedelta(hours=24)]
        candidates = sorted(
            (m for m in last_day_only if m.sender.automated and _removable(m)),
            key=lambda m: m.native_id,
        )
        for m in candidates[: n - high]:
            world.mails.remove(m)
            changed = True
    return changed


def _fix_events(world, day) -> bool:
    low, high = LIMITS["calendar"]
    snap = snapshot(world, day)
    n = len(snap.events)
    if n < low:
        for _ in range(low - n):
            world.add_filler_event(day)
        return True
    if n > high:
        candidates = [e for e in snap.events if e.filler] or [
            e for e in snap.events if _removable(e) and e.series_id is None and not e.all_day
        ]
        candidates.sort(
            key=lambda e: (-e.start.astimezone(PARIS).toordinal(), e.native_id)
        )  # le plus tardif d'abord
        for e in candidates[: n - high]:
            world.events.remove(e)
        return bool(candidates)
    return False


def _fix_tasks(world, day) -> bool:
    low, high = LIMITS["task"]
    snap = snapshot(world, day)
    n = len(snap.tasks)
    if n > high:
        catch_up = snap.now - timedelta(hours=15)  # la veille vers 17h30
        overdue = [
            t
            for t in snap.tasks
            if t.due_at is not None and t.due_at < snap.now and t.created_at < catch_up
        ]
        overdue.sort(
            key=lambda t: (not t.filler, "task_no_due" in t.tags, t.salience, t.due_at, t.native_id)
        )
        for t in overdue[: n - high]:
            t.completed_at = catch_up
        return bool(overdue)
    if n < low:
        for index in range(low - n):
            _filler_task(world, day, index)
        return True
    return False


_FILLER_TASKS = (
    ("Valider les notes de frais de l'équipe", "Portail RH"),
    ("Signer les bons de commande fournitures", "Assistante"),
    ("Répondre au sondage qualité interne", "Lien reçu par mail"),
    ("Mettre à jour le planning d'astreinte", "Fichier Planning"),
    ("Relancer le service informatique (imprimante)", "Ticket ouvert"),
)


def _filler_task(world, day, index: int) -> None:
    from collect.synth.clock import at

    prev = day - timedelta(days=1)
    while not is_working_day(prev):
        prev -= timedelta(days=1)
    title, notes = _FILLER_TASKS[(day.toordinal() + index) % len(_FILLER_TASKS)]
    t = world.task(title, notes, at(prev, 16, 10 + index), at(day, 17), salience=15, diligence=0.0)
    t.completed_at = at(day, 15, 30)
    t.filler = True


def _fix_files(world, day) -> bool:
    low, high = LIMITS["file"]
    snap = snapshot(world, day)
    n = len(snap.files)
    if n < low:
        for _ in range(low - n):
            world.add_filler_file(day)
        return True
    if n > high:
        candidates = [f for f, _v in snap.files if _removable(f) and f.salience <= 6]
        candidates.sort(key=lambda f: (not f.filler, f.native_id))
        for f in candidates[: n - high]:
            world.files.remove(f)
        return bool(candidates)
    return False
