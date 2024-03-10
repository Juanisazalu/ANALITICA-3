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
SELECT retirementDate, EmployeeID, Attrition FROM info_retiros
WHERE strftime('%Y',retirementDate) = '2016' AND 
retirementType = "Resignation" 
;
--Se crea la tabla para generar la condicion en where exists
DROP TABLE IF EXISTS info_retiros2_1 ;
CREATE TABLE info_retiros2_1 AS
SELECT retirementDate, EmployeeID, Attrition FROM info_retiros
WHERE  strftime('%Y',retirementDate) = '2015' AND 
retirementType = "Resignation" 
;

DROP TABLE IF EXISTS tabla_completa;
CREATE TABLE  tabla_completa AS
SELECT * FROM general_data2 AS gd
LEFT JOIN encuesta_empleado2 as ee ON gd.EmployeeID = ee.EmployeeID
LEFT JOIN encuesta_gerente2 as eg ON eg.EmployeeID = gd.EmployeeID
LEFT JOIN info_retiros2 AS ir ON ir.EmployeeID = eg.EmployeeID
;

DELETE FROM tabla_completa WHERE EXISTS (SELECT * FROM info_retiros2_1 AS ir2 
  WHERE ir2.EmployeeID = tabla_completa.EmployeeID);
--eliminar variables-


ALTER TABLE tabla_completa DROP COLUMN "index";
ALTER TABLE tabla_completa DROP COLUMN "index:1";
ALTER TABLE tabla_completa DROP COLUMN "index:2";
--ALTER TABLE tabla_completa DROP COLUMN 'index:3';
ALTER TABLE tabla_completa DROP COLUMN "EmployeeID:1";
ALTER TABLE tabla_completa DROP COLUMN "EmployeeID:2";
ALTER TABLE tabla_completa DROP COLUMN "EmployeeID:3";

ALTER TABLE tabla_completa ADD COLUMN v_objetivo INT;

-- Actualizar los valores de la nueva columna basados en la condición
UPDATE tabla_completa
SET v_objetivo = CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END;

--SELECT *,
--CASE WHEN Attrition = "Yes" THEN 1 ELSE 0 END AS v_obtejivo
--FROM tabla_completa;

ALTER TABLE tabla_completa DROP COLUMN 'Attrition';
ALTER TABLE tabla_completa DROP COLUMN 'infodate';
ALTER TABLE tabla_completa DROP COLUMN 'datesurvey';
ALTER TABLE tabla_completa DROP COLUMN 'surveydate';
ALTER TABLE tabla_completa DROP COLUMN 'retirementdate';

-------------------X TEST--------------------------------
DROP TABLE IF EXISTS general_data3;
CREATE TABLE general_data3 AS
SELECT * FROM general_data
WHERE  strftime('%Y',InfoDate) = '2016' 
;

DROP TABLE IF EXISTS encuesta_empleado3;
CREATE TABLE encuesta_empleado3 AS
SELECT * FROM encuesta_empleado
WHERE strftime('%Y',DateSurvey) = '2016'
;

DROP TABLE IF EXISTS encuesta_gerente3;
CREATE TABLE encuesta_gerente3 AS
SELECT * FROM encuesta_gerente
WHERE strftime('%Y',SurveyDate) = '2016'
;


DROP TABLE IF EXISTS tabla_completa2;
CREATE TABLE  tabla_completa2 AS
SELECT * FROM general_data2 AS gd2
LEFT JOIN encuesta_empleado2 as ee2 ON gd2.EmployeeID = ee2.EmployeeID
LEFT JOIN encuesta_gerente2 as eg2 ON eg2.EmployeeID = gd2.EmployeeID
;

DELETE FROM tabla_completa2 WHERE EXISTS (SELECT * FROM info_retiros AS ire 
  WHERE ire.EmployeeID = tabla_completa2.EmployeeID);
--eliminar variables-


ALTER TABLE tabla_completa2 DROP COLUMN "index";--
ALTER TABLE tabla_completa2 DROP COLUMN "index:1";
ALTER TABLE tabla_completa2 DROP COLUMN "index:2";
ALTER TABLE tabla_completa2 DROP COLUMN "EmployeeID:1";
ALTER TABLE tabla_completa2 DROP COLUMN "EmployeeID:2";
--ALTER TABLE tabla_completa2 DROP COLUMN "EmployeeID";--
ALTER TABLE tabla_completa2 DROP COLUMN 'infodate';
ALTER TABLE tabla_completa2 DROP COLUMN 'datesurvey';
ALTER TABLE tabla_completa2 DROP COLUMN 'surveydate';
ALTER TABLE tabla_completa2 DROP COLUMN 'yearsatcompany';
ALTER TABLE tabla_completa2 DROP COLUMN 'totalworkingyears';
ALTER TABLE tabla_completa2 DROP COLUMN 'performancerating';
ALTER TABLE tabla_completa2 DROP COLUMN 'gender';
