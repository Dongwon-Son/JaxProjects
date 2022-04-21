(define (domain blocksworld)
  (:requirements :strips :negative-preconditions)
  (:predicates (leftover ?x)
               (on-table ?x)
               (arm-empty)
               (holding ?x)
               (on ?x ?y)
               (box-empty)
               (top ?x)
               (bigger ?x ?y))

  (:action pickup
    :parameters (?ob)
    :precondition (and (top ?ob) (on-table ?ob) (arm-empty))
    :effect (and (holding ?ob) (not (top ?ob)) (not (on-table ?ob))
                 (not (arm-empty))))

  (:action placetable
    :parameters  (?ob)
    :precondition (and (holding ?ob))
    :effect (and (top ?ob) (arm-empty) (on-table ?ob)
                 (not (holding ?ob))))

  (:action placebox
      :parameters (?ob)
      :precondition (and (holding ?ob) (box-empty))
      :effect (and (not (box-empty)) (top ?ob) (not (holding ?ob)) (arm-empty))
  )

  (:action clearleftover
      :parameters (?ob)
      :precondition (and (holding ?ob) (leftover ?ob))
      :effect (and (not (leftover ?ob)))
  )

  (:action stack
    :parameters  (?ob ?underob)
    :precondition (and  (top ?underob) (holding ?ob) (bigger ?underob ?ob) (not (leftover ?underob)))
    :effect (and (arm-empty) (top ?ob) (on ?ob ?underob)
                 (not (top ?underob)) (not (holding ?ob))))

  (:action unstack
    :parameters  (?ob ?underob)
    :precondition (and (on ?ob ?underob) (top ?ob) (arm-empty))
    :effect (and (holding ?ob) (top ?underob)
                 (not (on ?ob ?underob)) (not (top ?ob)) (not (arm-empty)))))