DROP TABLE IF EXISTS general_data2;
CREATE TABLE general_data2 AS
SELECT * FROM general_data
WHERE  strftime('%Y',InfoDate) = '2015' 
;

DROP TABLE IF EXISTS encuesta_empleado2;
CREATE TABLE encuesta_empleado2 AS
SELECT * FROM encuesta_empleado
WHERE strftime('%Y',DateSurvey) = '2015'
;

DROP TABLE IF EXISTS encuesta_gerente2;
CREATE TABLE encuesta_gerente2 AS
SELECT * FROM encuesta_gerente
WHERE strftime('%Y',SurveyDate) = '2015'
;

DROP TABLE IF EXISTS info_retiros2 ;
CREATE TABLE info_retiros2 AS
SELECT * FROM info_retiros
WHERE strftime('%Y',retirementDate) = '2015'
;

DROP TABLE IF EXISTS tabla_completa;
CREATE TABLE  tabla_completa AS
SELECT * FROM general_data2 AS gd
LEFT JOIN encuesta_empleado2 as ee ON gd.EmployeeID = ee.EmployeeID
LEFT JOIN encuesta_gerente2 as eg ON eg.EmployeeID = gd.EmployeeID
LEFT JOIN info_retiros2 AS ir ON ir.EmployeeID = eg.EmployeeID
;

--eliminar variables-
--volver el nombre de las columnas minusculas
--rellenar nulos--

ALTER TABLE tabla_completa DROP COLUMN 'index';
ALTER TABLE tabla_completa DROP COLUMN 'index:1';
ALTER TABLE tabla_completa DROP COLUMN 'index:2';
ALTER TABLE tabla_completa DROP COLUMN 'index:3';
ALTER TABLE tabla_completa DROP COLUMN 'EmployeeID:1';
ALTER TABLE tabla_completa DROP COLUMN 'EmployeeID:2';
ALTER TABLE tabla_completa DROP COLUMN 'EmployeeID:3';

ALTER TABLE tabla_completa ADD COLUMN v_objetivo INT;

-- Actualizar los valores de la nueva columna basados en la condición
UPDATE tabla_completa
SET v_objetivo = CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END;

--SELECT *,
--CASE WHEN Attrition = "Yes" THEN 1 ELSE 0 END AS v_obtejivo
--FROM tabla_completa;

ALTER TABLE tabla_completa DROP COLUMN 'Attrition';