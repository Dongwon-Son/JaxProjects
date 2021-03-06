(define (problem dish_stack)
   (:domain blocksworld)
   (:objects a b c)
   (:init
     (on-table a)
     (on b a)
     (on-table c)
     (top b)
     (top c)
     (bigger a b)
     (bigger c b)
     (bigger a c)
     (leftover a)
     (leftover c)
     (arm-empty)
     (box-empty))
   (:goal
      ( and (not (on-table c)) (not (on-table a)) (not (on-table b)) (arm-empty))
   )
)