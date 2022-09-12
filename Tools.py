import sympy as sym

def DisEuLa(L,X,X_dot,F):
    """
    DisEuLa
    --------

    Parametros
    ----------
    L --> Lagrangiano expresión simbolica de sympy
    X --> lista con las variables simbolicas del vector de estado
    X_dot --> lista con las variables de las derivadas del vector de estado
    F ---> Lista con las fuerzas aplicadas a cada grado de libertad

    Ejemplo:
    import sympy as sym
    import DiscreteTolls as EL
    m,M,x,v_x,a_x,theta,omega,alpha,l,g,u,delta = sym.symbols("m,M,x,v_x,a_x,theta,omega,alpha,l,g,u,delta")
    T = 1/2*M*v_x**2 + 1/2*m*((v_x + l*omega*sym.cos(theta))**2 + (l*omega*sym.sin(theta))**2)
    V = -m*l*g*sym.cos(theta)
    L = T-V

    Sol = EL.Eu_La(L,[x,v_x,theta,omega],[v_x,a_x,omega,alpha],[u-delta*v_x,0])
    print(Sol)
    >>[Eq(-delta*v_x + u, 1.0*a_x*(M + m) + 1.0*alpha*l*m*cos(theta) - 1.0*l*m*omega**2*sin(theta)), Eq(l*m*(1.0*a_x*cos(theta) + alpha*l + g*sin(theta)), 0)]
    """
    L_M = sym.Matrix([L])
    X_M = sym.Matrix([X])
    X_dot_M = sym.Matrix([X_dot])
    jac = L_M.jacobian(X_M)
    ddt_jac = jac.jacobian(X_M)*X_dot_M.T
  
    #Ecuación Euler lagrange:
    RTA = []
    cont = 0
    for i in X_dot:
        if X.count(i)>0:
            Xind = X.index(i)
            X_dind = X_dot.index(i)
            RTA.append(sym.Eq(ddt_jac[Xind] - jac[X_dind],F[cont]).simplify())
            cont += 1
    return RTA

def linearize_fuction(f,X,X0):
    """
    Esta función linealiza haciendo uso de las series de tylor

    Parametros
    ----------

    f --> expresión simbolica de sympy de la ecuación a linealizar 
    X --> lista de variables en la ecuación
    X0 --> Punto de linealización, es una lista con los valores numéricos de dicho punto
    """

    m = len(X)
    pares = []   
    for i in range(m):
        pares.append((X[i],X0[i]))
    y = f.subs(pares)
    for i in range(m):
        y = y + sym.diff(f,X[i]).subs(pares)*(X[i] - X0[i])
    return y

def linearize_system(sis,X,X0):
    sis_lin = {}
    for i in sis.keys():
        sis_lin.__setitem__(i,linearize_fuction(sis[i],X,X0))
    return sis_lin
 

