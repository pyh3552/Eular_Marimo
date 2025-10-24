import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import sympy as sp
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np, sp


@app.cell
def _(sp):
    # 定义符号变量
    Theta_x, Theta_y, Theta_z = sp.symbols('Theta_x Theta_y Theta_z')
    return Theta_x, Theta_y, Theta_z


@app.cell
def _(sp):
    # 定义基本旋转矩阵
    def rotation_x(angle):
        return sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(angle), -sp.sin(angle)],
            [0, sp.sin(angle), sp.cos(angle)]
        ])

    def rotation_y(angle):
        return sp.Matrix([
            [sp.cos(angle), 0, sp.sin(angle)],
            [0, 1, 0],
            [-sp.sin(angle), 0, sp.cos(angle)]
        ])

    def rotation_z(angle):
        return sp.Matrix([
            [sp.cos(angle), -sp.sin(angle), 0],
            [sp.sin(angle), sp.cos(angle), 0],
            [0, 0, 1]
        ])
    return rotation_x, rotation_y, rotation_z


@app.cell
def _(rotation_x, rotation_y, rotation_z):
    # 构建ZYX顺序的欧拉角旋转矩阵
    def rotation_matrix_zyx(Theta_x, Theta_y, Theta_z):
        return  rotation_x(Theta_x) * rotation_y(Theta_y) * rotation_z(Theta_z)
    return (rotation_matrix_zyx,)


@app.cell
def _(Theta_x, Theta_y, Theta_z, rotation_matrix_zyx, sp):
    # 计算符号形式的旋转矩阵
    R_symbolic = rotation_matrix_zyx(Theta_x, Theta_y, Theta_z)
    R_symbolic_simplified = sp.simplify(R_symbolic)
    return R_symbolic, R_symbolic_simplified


@app.cell
def _(R_symbolic_simplified, mo, sp):
    # 显示符号形式的旋转矩阵
    mo.md(f"""## 欧拉角旋转矩阵（ZYX顺序）的符号表达式
    $${sp.latex(R_symbolic_simplified)}$$
    """
    )
    return


@app.cell
def _(mo):
    # 定义UI滑块
    Theta_x_slider = mo.ui.slider(0, 360, 1, label="Roll x轴 (度)")
    Theta_y_slider = mo.ui.slider(0, 360, 10, label="Pitch y轴 (度)")
    Theta_z_slider = mo.ui.slider(0, 360, 1, label="Yaw z轴 (度)")
    return Theta_x_slider, Theta_y_slider, Theta_z_slider


@app.cell
def _(R_symbolic_simplified, Theta_y, Theta_y_slider, sp):
    Y_90_Rotmat = R_symbolic_simplified.subs(Theta_y, sp.pi*Theta_y_slider.value/180.0)
    return (Y_90_Rotmat,)


@app.cell
def _(Theta_y_slider, Y_90_Rotmat, mo, sp):
    # 显示符号形式的旋转矩阵
    mo.md(f"""## 当绕Y轴的旋转为{Theta_y_slider.value}°时，欧拉角旋转矩阵（ZYX顺序）的符号表达式
    $${sp.latex(Y_90_Rotmat)}$$
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(Theta_x_slider, Theta_y_slider, Theta_z_slider, mo):
    # 显示滑块
    mo.md(f"""## 欧拉角控制面板
    {Theta_x_slider}
    {Theta_y_slider}
    {Theta_z_slider}""")
    return


@app.cell
def _(Theta_x_slider, Theta_y_slider, Theta_z_slider, np):
    # 获取当前角度值（弧度）
    X_rad = np.radians(Theta_x_slider.value)
    Y_rad = np.radians(Theta_y_slider.value)
    Z_rad = np.radians(Theta_z_slider.value)
    return X_rad, Y_rad, Z_rad


@app.cell
def _(R_symbolic, Theta_x, Theta_y, Theta_z, np):
    # 计算数值旋转矩阵
    def calculate_rotation_matrix(X_rad, Y_rad, Z_rad):
        # R_numeric = R_symbolic.subs([(Theta_z, Z_rad), (Theta_y, Y_rad), (Theta_x, X_rad)])
        R_numeric = R_symbolic.subs([(Theta_x, X_rad), (Theta_z, Z_rad), (Theta_y, Y_rad), ])
        return np.array(R_numeric).astype(float)
    return (calculate_rotation_matrix,)


@app.cell
def _(X_rad, Y_rad, Z_rad, calculate_rotation_matrix):
    R = calculate_rotation_matrix(X_rad, Y_rad, Z_rad)
    return (R,)


@app.function
# 检测是否接近万向节死锁
def check_gimbal_lock(beta_degrees):
    threshold = 5  # 接近死锁的阈值（度）
    if abs(beta_degrees - 90) < threshold:
        return True, f"⚠️ 警告：接近万向节死锁！β = {beta_degrees}°接近90°"
    return False, ""


@app.cell
def _(Theta_y_slider):
    # 检查当前状态
    is_locked, lock_message = check_gimbal_lock(Theta_y_slider.value)
    return is_locked, lock_message


@app.cell
def _(is_locked, lock_message, mo):
    # 显示警告（如果需要）
    if is_locked:
        mo.md(lock_message)
    else:
        mo.md("✅ 当前状态：正常旋转（无万向节死锁）")
    return


@app.cell
def _(R, mo, np):
    # 显示当前旋转矩阵
    mo.md(f"""## 当前旋转矩阵
    ```
    {np.round(R, 4)}
    ```""")
    return


@app.cell
def _(go, make_subplots, np):
    # 可视化函数
    def visualize_rotation(R):
        # 原始坐标轴
        origin = np.array([0, 0, 0])
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # 旋转后的坐标轴
        x_rotated = R @ x_axis
        y_rotated = R @ y_axis
        z_rotated = R @ z_axis

        # 创建图形
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                            subplot_titles=("原始坐标系", "旋转后的坐标系"))

        # 原始坐标系
        # X轴（红色）
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], x_axis[0]],
                y=[origin[1], x_axis[1]],
                z=[origin[2], x_axis[2]],
                mode='lines',
                line=dict(color='red', width=5),
                name='X轴'
            ),
            row=1, col=1
        )

        # Y轴（绿色）
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], y_axis[0]],
                y=[origin[1], y_axis[1]],
                z=[origin[2], y_axis[2]],
                mode='lines',
                line=dict(color='green', width=5),
                name='Y轴'
            ),
            row=1, col=1
        )

        # Z轴（蓝色）
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], z_axis[0]],
                y=[origin[1], z_axis[1]],
                z=[origin[2], z_axis[2]],
                mode='lines',
                line=dict(color='blue', width=5),
                name='Z轴'
            ),
            row=1, col=1
        )

        # 旋转后的坐标系
        # 旋转后的X轴（红色）
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], x_rotated[0]],
                y=[origin[1], x_rotated[1]],
                z=[origin[2], x_rotated[2]],
                mode='lines',
                line=dict(color='red', width=5),
                name='旋转后X轴'
            ),
            row=1, col=2
        )

        # 旋转后的Y轴（绿色）
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], y_rotated[0]],
                y=[origin[1], y_rotated[1]],
                z=[origin[2], y_rotated[2]],
                mode='lines',
                line=dict(color='green', width=5),
                name='旋转后Y轴'
            ),
            row=1, col=2
        )

        # 旋转后的Z轴（蓝色）
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], z_rotated[0]],
                y=[origin[1], z_rotated[1]],
                z=[origin[2], z_rotated[2]],
                mode='lines',
                line=dict(color='blue', width=5),
                name='旋转后Z轴'
            ),
            row=1, col=2
        )

        # 更新布局
        fig.update_layout(
            height=500,
            width=900,
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[-1.2, 1.2]),
                yaxis=dict(range=[-1.2, 1.2]),
                zaxis=dict(range=[-1.2, 1.2])
            ),
            scene2=dict(
                aspectmode='cube',
                xaxis=dict(range=[-1.2, 1.2]),
                yaxis=dict(range=[-1.2, 1.2]),
                zaxis=dict(range=[-1.2, 1.2])
            )
        )

        return fig
    return (visualize_rotation,)


@app.cell
def _(R, visualize_rotation):
    # 创建可视化
    fig = visualize_rotation(R)
    fig
    return


@app.cell
def _(Theta_x_slider, Theta_z_slider, mo):
    # 万向节死锁分析
    mo.md(f"""## 万向节死锁分析

    在ZYX旋转顺序中，当β = 90°时，旋转矩阵只依赖于γ-α的差值，而不是它们的各自值。

    当前Z轴 = {Theta_z_slider.value:.1f}°, X轴 = {Theta_x_slider.value:.1f}°

    **差值 γ-α = {Theta_x_slider.value - Theta_z_slider.value:.1f}°**

    如果β接近90°，尝试改变α和γ，但保持它们的差值不变，你会发现旋转效果相同。这就是万向节死锁的本质：丢失了一个旋转自由度。""")
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
