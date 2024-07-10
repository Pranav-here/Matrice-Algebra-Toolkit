import numpy as np
from scipy.linalg import lu, cholesky, qr, svd, hessenberg


def reduced_row_echelon_form(matrix):
    M = matrix.copy()
    lead = 0
    rows, cols = M.shape

    for r in range(rows):
        if lead >= cols:
            break
        i = r
        while M[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    return M
        M[[i, r]] = M[[r, i]]
        lv = M[r, lead]
        M[r] = M[r] / lv
        for i in range(rows):
            if i != r:
                M[i] -= M[i, lead] * M[r]
        lead += 1

    return M


def row_echelon_form(matrix):
    rref = reduced_row_echelon_form(matrix)
    rre = rref[:, rref.any(0)]
    return rre


def EigenValues():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print("Eigenvalues:")
    print(eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)


def Determinant():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    det = np.linalg.det(matrix)
    print("Determinant:")
    print(det)


def Rank():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    rank = np.linalg.matrix_rank(matrix)
    print("Rank:")
    print(rank)


def Trace():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    trace = np.trace(matrix)
    print("Trace:")
    print(trace)


def MatrixExponential():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    exp_matrix = np.linalg.expm(matrix)
    print("Matrix Exponential:")
    print(exp_matrix)


def LUDecomposition():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    P, L, U = lu(matrix)
    print("LU Decomposition:")
    print("P:")
    print(P)
    print("L:")
    print(L)
    print("U:")
    print(U)


def CholeskyDecomposition():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    L = cholesky(matrix, lower=True)
    print("Cholesky Decomposition (L):")
    print(L)


def SolveLinearSystem():
    dimensions = int(input("What are the dimensions of your coefficient matrix (e.g., for a 2x2 matrix, enter 2): "))
    A = np.zeros((dimensions, dimensions))
    b = np.zeros(dimensions)

    print("Enter the elements of the coefficient matrix A:")
    for i in range(dimensions):
        for j in range(dimensions):
            A[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    print("Enter the elements of the constant matrix b:")
    for i in range(dimensions):
        b[i] = float(input(f"Element ({i+1}): "))

    solution = np.linalg.solve(A, b)
    print("Solution of the linear system (Ax = b):")
    print(solution)


def Norm():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    frobenius_norm = np.linalg.norm(matrix, 'fro')
    inf_norm = np.linalg.norm(matrix, np.inf)
    print("Frobenius Norm:")
    print(frobenius_norm)
    print("Infinity Norm:")
    print(inf_norm)


def QRDecomposition():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    Q, R = qr(matrix)
    print("QR Decomposition:")
    print("Q:")
    print(Q)
    print("R:")
    print(R)


def SingularValueDecomposition():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    U, s, Vh = svd(matrix)
    print("Singular Value Decomposition:")
    print("U:")
    print(U)
    print("s:")
    print(s)
    print("Vh:")
    print(Vh)


def MatrixPower():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    power = int(input("Enter the power to raise the matrix to: "))
    result = np.linalg.matrix_power(matrix, power)
    print(f"Matrix raised to the power {power}:")
    print(result)


def ConditionNumber():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    condition_number = np.linalg.cond(matrix)
    print("Condition Number:")
    print(condition_number)


def MoorePenrosePseudoinverse():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    pseudoinverse = np.linalg.pinv(matrix)
    print("Moore-Penrose Pseudoinverse:")
    print(pseudoinverse)


def Projection():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))
    vector = np.zeros(dimensions)

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    print("Enter the elements of the vector:")
    for i in range(dimensions):
        vector[i] = float(input(f"Element ({i+1}): "))

    projection = np.dot(matrix, vector) / np.dot(vector, vector) * vector
    print("Projection of the vector onto the matrix:")
    print(projection)


def GramSchmidtProcess():
    dimensions = int(input("How many vectors do you have (e.g., for 2 vectors, enter 2): "))
    vectors = np.zeros((dimensions, dimensions))

    print("Enter the elements of the vectors:")
    for i in range(dimensions):
        for j in range(dimensions):
            vectors[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    orthogonal_vectors = []
    for i in range(dimensions):
        v = vectors[i]
        for j in range(i):
            v = v - np.dot(vectors[i], orthogonal_vectors[j]) * orthogonal_vectors[j]
        orthogonal_vectors.append(v / np.linalg.norm(v))

    print("Orthogonalized Vectors using Gram-Schmidt Process:")
    print(np.array(orthogonal_vectors))


def DeterminantOfSubmatrices():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 3x3 matrix, enter 3): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    submatrices_determinants = []
    for i in range(1, dimensions + 1):
        for j in range(1, dimensions + 1):
            submatrix = matrix[:i, :j]
            det = np.linalg.det(submatrix)
            submatrices_determinants.append(det)

    print("Determinants of Submatrices:")
    print(submatrices_determinants)


def HessenbergForm():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 3x3 matrix, enter 3): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    H = hessenberg(matrix)
    print("Hessenberg Form:")
    print(H)


def JordanCanonicalForm():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 3x3 matrix, enter 3): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    eigvals, eigvecs = np.linalg.eig(matrix)
    J = np.diag(eigvals)
    P = eigvecs
    Pinv = np.linalg.inv(P)
    JCF = P @ J @ Pinv
    print("Jordan Canonical Form:")
    print(JCF)


def MatrixOperations():
    n = int(input("How many matrices do you have: "))
    operation = input("Choose operation:\nAdd: +\nSubtract: -\nScalar Multiplication: *\nDot Product: .\nCross Product: x\nTranspose: T\nInverse: I\nSymmetric: S\nSkew-Symmetric: Sk\nRREF: 11\nRRE: 12\n").strip().lower()

    matrices = []

    for _ in range(n):
        a = int(input("Size of array (e.g., for a 2x2 matrix, enter 2): "))
        matrix = np.zeros((a, a))

        print(f"Enter elements for matrix {_ + 1}:")
        for i in range(a):
            for j in range(a):
                matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

        matrices.append(matrix)

    if operation == "+":
        result = sum(matrices)
    elif operation == "-":
        result = matrices[0] - sum(matrices[1:])
    elif operation == "*":
        scalar = float(input("Enter the scalar value: "))
        result = [matrix * scalar for matrix in matrices]
    elif operation == ".":
        if n != 2:
            print("Dot product requires exactly 2 matrices.")
            return
        result = np.dot(matrices[0], matrices[1])
    elif operation == "x":
        if n != 2:
            print("Cross product requires exactly 2 matrices.")
            return
        result = np.cross(matrices[0], matrices[1])
    elif operation == "t":
        result = [np.transpose(matrix) for matrix in matrices]
    elif operation == "i":
        result = [np.linalg.inv(matrix) for matrix in matrices]
    elif operation == "s":
        result = [matrix for matrix in matrices if np.allclose(matrix, np.transpose(matrix))]
    elif operation == "sk":
        result = [matrix for matrix in matrices if np.allclose(matrix, -np.transpose(matrix))]
    elif operation == "11":
        result = [reduced_row_echelon_form(matrix) for matrix in matrices]
    elif operation == "12":
        result = [row_echelon_form(matrix) for matrix in matrices]
    else:
        print("Invalid operation.")
        return

    print("Result:")
    print(result)


def PolarDecomposition():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    U, H = np.linalg.polar(matrix)
    print("Polar Decomposition:")
    print("U (Unitary Matrix):")
    print(U)
    print("H (Positive-Semidefinite Hermitian Matrix):")
    print(H)


def PositiveDefiniteCheck():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 2x2 matrix, enter 2): "))
    matrix = np.zeros((dimensions, dimensions))

    print("Enter the elements of the matrix:")
    for i in range(dimensions):
        for j in range(dimensions):
            matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))

    try:
        np.linalg.cholesky(matrix)
        print("The matrix is positive definite.")
    except np.linalg.LinAlgError:
        print("The matrix is not positive definite.")


def ToeplitzMatrix():
    dimensions = int(input("What are the dimensions of your matrix (e.g., for a 3x3 matrix, enter 3): "))
    first_row = []
    first_column = []

    print("Enter the elements of the first row:")
    for i in range(dimensions):
        first_row.append(float(input(f"Element ({1},{i+1}): ")))

    print("Enter the elements of the first column:")
    for i in range(1, dimensions):
        first_column.append(float(input(f"Element ({i+1},{1}): ")))

    first_column.insert(0, first_row[0])
    toeplitz_matrix = np.zeros((dimensions, dimensions))
    for i in range(dimensions):
        toeplitz_matrix[i, :] = np.roll(first_row, i)
        toeplitz_matrix[:, i] = np.roll(first_column, i)

    print("Toeplitz Matrix:")
    print(toeplitz_matrix)


def menu():
    operations = {
        '1': 'Matrix Operations',
        '2': 'Eigenvalues and Eigenvectors',
        '3': 'Determinant',
        '4': 'Rank',
        '5': 'Trace',
        '6': 'Matrix Exponential',
        '7': 'LU Decomposition',
        '8': 'Cholesky Decomposition',
        '9': 'Solve Linear System',
        '10': 'Matrix Norms',
        '11': 'QR Decomposition',
        '12': 'Singular Value Decomposition',
        '13': 'Matrix Power',
        '14': 'Condition Number',
        '15': 'Moore-Penrose Pseudoinverse',
        '16': 'Projection',
        '17': 'Gram-Schmidt Process',
        '18': 'Determinant Of Submatrices',
        '19': 'Hessenberg Form',
        '20': 'Jordan Canonical Form',
        '21': 'Reduced Row Echelon Form (RREF)',
        '22': 'Row Echelon Form (REF)',
        '23': 'Polar Decomposition',
        '24': 'Positive Definite Check',
        '25': 'Toeplitz Matrix'
    }

    print("Choose an operation by entering the corresponding number:")
    for num, operation in operations.items():
        print(f"{num}. {operation}")

    choice = input("Enter your choice (1-25): ")

    if choice == '1':
        MatrixOperations()
    elif choice == '2':
        EigenValues()
    elif choice == '3':
        Determinant()
    elif choice == '4':
        Rank()
    elif choice == '5':
        Trace()
    elif choice == '6':
        MatrixExponential()
    elif choice == '7':
        LUDecomposition()
    elif choice == '8':
        CholeskyDecomposition()
    elif choice == '9':
        SolveLinearSystem()
    elif choice == '10':
        Norm()
    elif choice == '11':
        QRDecomposition()
    elif choice == '12':
        SingularValueDecomposition()
    elif choice == '13':
        MatrixPower()
    elif choice == '14':
        ConditionNumber()
    elif choice == '15':
        MoorePenrosePseudoinverse()
    elif choice == '16':
        Projection()
    elif choice == '17':
        GramSchmidtProcess()
    elif choice == '18':
        DeterminantOfSubmatrices()
    elif choice == '19':
        HessenbergForm()
    elif choice == '20':
        JordanCanonicalForm()
    elif choice == '21':
        dimensions = int(input("Enter the number of rows: "))
        cols = int(input("Enter the number of columns: "))
        matrix = np.zeros((dimensions, cols))
        print("Enter the elements of the matrix:")
        for i in range(dimensions):
            for j in range(cols):
                matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))
        result = reduced_row_echelon_form(matrix)
        print("Reduced Row Echelon Form (RREF):")
        print(result)
    elif choice == '22':
        dimensions = int(input("Enter the number of rows: "))
        cols = int(input("Enter the number of columns: "))
        matrix = np.zeros((dimensions, cols))
        print("Enter the elements of the matrix:")
        for i in range(dimensions):
            for j in range(cols):
                matrix[i, j] = float(input(f"Element ({i+1},{j+1}): "))
        result = row_echelon_form(matrix)
        print("Row Echelon Form (RRE):")
        print(result)
    elif choice == '23':
        PolarDecomposition()
    elif choice == '24':
        PositiveDefiniteCheck()
    elif choice == '25':
        ToeplitzMatrix()
    else:
        print("Invalid choice. Please enter a number between 1 and 25.")


if __name__ == "__main__":
    while True:
        menu()
        continue_choice = input("Do you want to perform another operation? (yes/no): ").strip().lower()
        if continue_choice != 'yes':
            print("Thank you!!!")
            break
